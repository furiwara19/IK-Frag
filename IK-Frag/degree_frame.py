"""
This routine converts scattering angles between the center-of-mass and laboratory frames.
While the CM → Lab transformation can be expressed analytically using arctan(),
the inverse transformation (Lab → CM) is non-trivial, as it depends significantly on the Lorentz gamma factor and the laboratory scattering angle.
Hence, the optimal CM angle is numerically determined using the optimization utilities of scipy.optimize.
"""

import numpy as np

# Convert cm frame from lab frame
def lab_to_cm_frame_theta(theta_L_deg, gamma, num_intervals=500):
    """
    Convert cm frame from lab frame

    Parameters:
        theta_L_deg (float): degree in lab frame
        gamma (float)
        num_intervals (int): precisely to find solution

    Returns:
        list of float: Degree list in CM frame
    """
        
    import numpy as np
    from scipy.optimize import root_scalar


    theta_L_rad = np.deg2rad(theta_L_deg)
    tan_theta_L = np.tan(theta_L_rad)

    def f(theta_c):
        return np.sin(theta_c) / (gamma + np.cos(theta_c)) - tan_theta_L

    theta_grid = np.linspace(0.001, np.pi - 0.001, num_intervals)
    solutions = []

    for i in range(len(theta_grid) - 1):
        a, b = theta_grid[i], theta_grid[i + 1]
        if f(a) * f(b) < 0:
            sol = root_scalar(f, bracket=[a, b], method='brentq')
            if sol.converged:
                root = sol.root
                if not any(np.isclose(root, s, atol=1e-5) for s in solutions):
                    solutions.append(root)

    return [np.rad2deg(s) for s in sorted(solutions)]


def cm_to_ex_theta(theta_c, gamma):
    """
    cm to lab frame
    tan(theta_L) = sin(theta_c)/(gamma + cos(theta_c))
    """
    import numpy as np

    theta_c_rad = np.radians(theta_c)

    theta_L = np.arctan(np.sin(theta_c_rad)/(gamma + np.cos(theta_c_rad)))

    return np.degrees(theta_L)


def DDX_cm_frame(E_incident, theta, sigma):
    """
    obtain DDX value for neutron incident energy and its scattered angle

    E_incident: energies of incident neutrons
    theta: Degree list to calculate DDX
    sigma: {
        "Energy": "energy": [E1, E2, ..., En],
        "DDX": [f(theta_1), f(theta_2), ..., f(theta_n)]
    }
    """

    if isinstance(sigma, dict):
        if np.isclose(sigma["Energy"], E_incident, rtol=1e-5):
            return sigma["DDX"](theta) 
        else:
            raise ValueError(f"Energy filling E_incident = {E_incident} couldn't be found")

def cm_to_lab_frame_DDX(gamma, theta_c, sigma_cm):
    """
    convert DDX value in CM frame into laboratory frame

    Parameters:
        gamma (float): gamma factor
        theta_c (float): degree list of scattered neutrons in CM frame
        E_em (float): energy list of scattered neutrons (MeV)
        sigma_cm (float): DDX value list in CM frame

    Returns:
        DDX_dict (dict): {"Energy": Energies of scattered neutrons
        "theta": emission degree of scattered neutrons (lab frame)
        "DDX": DDX values for each energy of incident neutrons and emission degree (lab frame)
        }
    """
    def cm_to_ex_theta(theta_c, gamma):
        theta_c_rad = np.radians(theta_c)
        theta_L = np.arctan(np.sin(theta_c_rad) / (gamma + np.cos(theta_c_rad)))
        return np.degrees(theta_L)

    theta_c_rad = np.radians(theta_c)

    numerator = (1 + gamma**2 + 2 * gamma * np.cos(theta_c_rad))**(3/2)
    denominator = np.abs(1 + gamma * np.cos(theta_c_rad))
    jacobian = numerator / denominator

    theta_lab = cm_to_ex_theta(theta_c, gamma)
    sigma_lab = jacobian * sigma_cm

    return sigma_lab
