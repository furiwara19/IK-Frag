# module import
import numpy as np
from scipy.optimize import root_scalar
from neutron_energy_angle import gamma_factor, energy_range_determination, maximum_angle, particles_energy, import_material_params
from degree_frame import cm_to_ex_theta, lab_to_cm_frame_theta, cm_to_lab_frame_DDX, DDX_cm_frame
from tqdm import tqdm
import bisect
from scipy.sparse import lil_matrix
import pandas as pd
import os

def _is_valid_gamma(g):
    """
    Minimal validity check for the gamma used in CM→Lab mapping.

    Requirements:
    - finite, not NaN/Inf
    - strictly > 1.0 (so that expressions like arcsin(1/gamma) remain defined)
      If your mapping allows gamma==1.0, relax this as needed.
    """
    return (g is not None) and np.isfinite(g) and (g > 1.0)

def merge_ddx(
    Material_params,
    E_incident,
    theta_lab,
    emission_frag_data,
    material,
    DDX_cm_ground,
    DDX_cm_excite,
    XS_ground,
    XS_excite,
    degree_step,
):
    """
    Build Lab-frame DDX matrices by merging ground and excited channels,
    with robust handling of kinematically closed excited states.

    Robustness note
    ---------------
    For some incident energies, the excited state can be kinematically closed
    or numerically ill-posed. In those cases, we mark the excited channel as
    'closed' for that energy and skip all excited-channel computations; the
    returned matrix is then formed from the ground channel only.
    """

    # --- External helpers expected in the environment ---
    # import_material_params -> (Ma, MA, Mb, MB, Q_ground, Q_excite)
    # gamma_factor(Ma,MA,Mb,MB,Q,E,material) -> float
    # maximum_angle(Ma,MA,Mb,MB,Q,E,material) -> float (deg in Lab)
    # cm_to_ex_theta(theta_cm_deg, gamma) -> theta_lab_deg
    # energy_range_determination(...) -> (En_g, En_x, status_dict)
    # lab_to_cm_frame_theta(theta_lab_deg, gamma, num_intervals=...) -> [theta_cm_deg,...]
    # cm_to_lab_frame_DDX(gamma, theta_cm_deg, ddx_cm_value) -> ddx_lab_value
    # particles_energy(theta_lab_deg, theta_Lab_map, E_n_map, emission_frag_data) -> energies

    Ma, MA, Mb, MB, Q_ground, Q_excite = import_material_params(Material_params)
    theta_cm = np.arange(0.0, 180.0 + degree_step, degree_step)

    DDX_matrixs = []
    theta_maps  = {"ground": [], "excite": []}  # last-energy diagnostics
    energy_maps = {"ground": [], "excite": []}  # last-energy diagnostics
    gammas      = {"ground": None, "excite": None}

    for i_e, e in enumerate(tqdm(E_incident, desc="DDX data generating...", unit="Energy")):
        if i_e == 0:
            # Preserve original behavior for the first slot
            DDX_matrixs.append(
                np.zeros((len(emission_frag_data), len(theta_lab)), dtype=np.float32)
            )
            continue

        # --- Kinematics (per incident energy) ---
        gamma_g = gamma_factor(Ma, MA, Mb, MB, Q_ground, e, material)

        # Compute excited-state gamma safely; treat as 'closed' if invalid
        excite_closed = False
        try:
            gamma_x = gamma_factor(Ma, MA, Mb, MB, Q_excite, e, material)
            if not _is_valid_gamma(gamma_x):
                excite_closed = True
                gamma_x = None
        except Exception:
            excite_closed = True
            gamma_x = None

        theta_max = maximum_angle(Ma, MA, Mb, MB, Q_ground, e, material)
        gammas = {"ground": gamma_g, "excite": (gamma_x if not excite_closed else None)}

        # CM→Lab maps sampled on theta_cm
        theta_Lab_g, theta_Lab_x = [], []
        E_n_g, E_n_x = [], []
        for theta_cm_deg in theta_cm:
            En_g = energy_range_determination(
                Ma, MA, Mb, MB, Q_ground, theta_cm_deg, e, material
            )
            En_x = energy_range_determination(
                Ma, MA, Mb, MB, Q_excite, theta_cm_deg, e, material
            )
            E_n_g.append(En_g)
            E_n_x.append(En_x)
            
            theta_Lab_g.append(cm_to_ex_theta(theta_cm_deg, gamma_g))
            # Only compute excited mapping if the channel is open
            if not excite_closed:
                theta_Lab_x.append(cm_to_ex_theta(theta_cm_deg, gamma_x))
            else:
                theta_Lab_x.append(np.nan)  # placeholder; not used downstream

        # Save last-energy diagnostics
        theta_maps  = {"ground": theta_Lab_g, "excite": theta_Lab_x}
        energy_maps = {"ground": E_n_g,     "excite": E_n_x}

        # DDX sources for this E
        e_key = round(float(e), 4)
        xs_g = float(XS_ground[i_e])
        xs_x = float(XS_excite[i_e])

        ddx_g_src = DDX_cm_ground.get(e_key, (lambda mu: 0.0))

        # If excited state is closed OR XS_excite==0, disable the excited channel
        if excite_closed or (xs_x == 0.0):
            ddx_x_src = 0
        else:
            ddx_x_src = DDX_cm_excite.get(e_key, (lambda mu: 0.0))

        # Per-angle evaluation
        energies_g_per_theta, values_g_per_theta = [], []
        energies_x_per_theta, values_x_per_theta = [], []
        sigma_sum_g, sigma_sum_x = 0.0, 0.0

        for theta_out in theta_lab:
            if theta_out > theta_max:
                energies_g_per_theta.append([])
                values_g_per_theta.append([])
                energies_x_per_theta.append([])
                values_x_per_theta.append([])
                continue

            # ground (always open by construction here)
            theta_cm_list_g = lab_to_cm_frame_theta(theta_out, gamma_g, num_intervals=500)
            Eg = particles_energy(theta_out, theta_Lab_g, E_n_g, emission_frag_data)

            Vg = []
            for deg_cm in theta_cm_list_g:
                mu = float(np.cos(np.radians(np.float32(deg_cm))))
                v_cm = ddx_g_src(mu) if callable(ddx_g_src) else ddx_g_src.subs({'mu': mu}).evalf()
                v_lab = cm_to_lab_frame_DDX(gamma_g, deg_cm, v_cm)
                val = float(v_lab)
                Vg.append(val)
                sigma_sum_g += val

            energies_g_per_theta.append(np.atleast_1d(Eg))
            values_g_per_theta.append(np.atleast_1d(Vg))

            # excited (only if open & source available)
            if ddx_x_src != 0 and not excite_closed:
                theta_cm_list_x = lab_to_cm_frame_theta(theta_out, gamma_x, num_intervals=500)
                Ex = particles_energy(theta_out, theta_Lab_x, E_n_x, emission_frag_data)

                Vx = []
                for deg_cm in theta_cm_list_x:
                    mu = float(np.cos(np.radians(np.float32(deg_cm))))
                    v_cm = ddx_x_src(mu) if callable(ddx_x_src) else ddx_x_src.subs({'mu': mu}).evalf()
                    v_lab = cm_to_lab_frame_DDX(gamma_x, deg_cm, v_cm)
                    val = float(v_lab)
                    Vx.append(val)
                    sigma_sum_x += val

                energies_x_per_theta.append(np.atleast_1d(Ex))
                values_x_per_theta.append(np.atleast_1d(Vx))
            else:
                energies_x_per_theta.append([])
                values_x_per_theta.append([])

        # Cross-section scaling for the excited channel
        if (xs_g == 0.0) or (sigma_sum_x == 0.0) or excite_closed:
            scale = 0.0
        else:
            scale = (sigma_sum_g / sigma_sum_x) * (xs_x / xs_g)

        # Populate sparse matrix (bin-wise addition)
        ddx_matrix = lil_matrix((len(emission_frag_data), len(theta_lab)), dtype=np.float32)

        for j, _theta_out in enumerate(theta_lab):
            # ground
            Eg, Vg = energies_g_per_theta[j], values_g_per_theta[j]
            if hasattr(Eg, "size") and hasattr(Vg, "size") and Eg.size and Vg.size:
                for energy, val in zip(Eg, Vg):
                    i = np.searchsorted(emission_frag_data, energy)
                    if 0 <= i < len(emission_frag_data):
                        ddx_matrix[i, j] += float(val)

            # excited (scaled), only if scale>0
            Ex, Vx = energies_x_per_theta[j], values_x_per_theta[j]
            if hasattr(Ex, "size") and hasattr(Vx, "size") and Ex.size and Vx.size and (scale != 0.0):
                for energy, val in zip(Ex, (Vx * scale)):
                    i = np.searchsorted(emission_frag_data, energy)
                    if 0 <= i < len(emission_frag_data):
                        ddx_matrix[i, j] += float(val)

        dense_ddx = ddx_matrix.toarray()

        DDX_matrixs.append(dense_ddx)

    os.makedirs("ddx_csv", exist_ok=True)

    # 例: 列は θ_lab、行は Emission energyl
    theta_cols = [float(t) for t in theta_lab]
    row_index  = [float(en) for en in emission_frag_data]

    for idx_e, (E, M) in enumerate(zip(E_incident, DDX_matrixs)):
        # M.shape = (len(emission_frag_data), len(theta_lab))
        df = pd.DataFrame(M, index=row_index, columns=theta_cols)
        df.index.name = "E_n [out]"
        df.columns.name = "theta_lab [deg]"
        # ファイル名に入射エネルギーを付与
        df.to_csv(f"ddx_csv/DDX_Einc_{float(E):.6g}.csv")

    return DDX_matrixs, theta_maps, energy_maps, gammas
