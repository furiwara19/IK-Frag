import re
import sympy as sp
from scipy.special import legendre
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_DDX(file_path, projectile, energy_threshold):
    """
    Load DDX values from nuclear data libraries.
    While ENDF includes the DDX value for the degree, 
    JENDL & TENDL contains it as the legendre function for degree.

    Parameters:
        file_path (str): slected file about nuclear data (dat file)

    Returns:
        tuple:
            - angle_distribution_ground (dict): ground state
            - angle_distribution_excite (dict): excite state
    """

    with open(file_path, errors='ignore') as f:
        lines = f.readlines()
        
    parts = [s.strip() for s in lines[0].split('/')]
    lib = parts[0]

    if projectile == "Li":
        marker_ground = "328 6 50"
        marker_excite = "328 6 51"
    else: # Be
        if lib == 'ENDF':
            marker_ground = "425 6 50"
            marker_excite = "425 6 51"  
        elif lib == 'JENDL':
            marker_ground = "409 6 50"
            marker_excite = "409 6 51" 
        else:
            marker_ground = "409 6 50"
            marker_excite = "409 6 51" 

    if "ENDF" in lib.upper() or "JENDL" in lib.upper():
        data_ground_raw = find_target_data(lines, marker_ground, threshold=energy_threshold)
        data_excite_raw = find_target_data(lines, marker_excite, threshold=energy_threshold)
        coeff_dict_ground = legendre_coefficients_processing(data_ground_raw)
        coeff_dict_excite = legendre_coefficients_processing(data_excite_raw)
        if  lib == 'ENDF' and projectile == "Li":
            DDX_ground = angular_distributions(coeff_dict_ground)
            DDX_excite = angular_distributions(coeff_dict_excite)
        else:
            DDX_ground = legendre_function_endf_format(coeff_dict_ground)
            DDX_excite = legendre_function_endf_format(coeff_dict_excite)
    else: # TENDL
        data_ground_raw = find_target_data(lines, marker_ground, threshold=energy_threshold)
        coeff_dict_ground = legendre_coefficients_processing(data_ground_raw)
        DDX_ground = legendre_function_endf_format(coeff_dict_ground)
        if projectile == "Li":
            DDX_excite = {key: 0 for key in DDX_ground.keys()}
        else:
            data_excite_raw = find_target_data(lines, marker_excite, threshold=energy_threshold)
            coeff_dict_excite = legendre_coefficients_processing(data_excite_raw)
            DDX_excite = legendre_function_endf_format(coeff_dict_excite)
    return DDX_ground, DDX_excite

def find_target_data(lines, marker, threshold, start_from=7):
    target = []
    reading = False
    counter = 0
    for line in lines:
        if marker in line:
            counter += 1
            if counter < start_from:
                continue
            elif counter == start_from:
                reading = True
        else:
            if not reading:
                continue

        row = [convert_scalar(tok) for tok in line.split()]

        if len(row) > 1 and isinstance(row[1], (int, float)) and row[1] > threshold:
            break
        if row[1] == 0:
            continue
        if marker in line:
            line_before_marker = line.split(marker)[0].strip()

            pattern = r'[-+]?\d*\.\d+(?:[eE]?[-+]?\d+)?'
            matches = re.findall(pattern, line_before_marker)

            row = [convert_scalar(tok) for tok in matches]

        if row:
            target.append(row)

    return target

def convert_scalar(s):
    import re
    t = s.strip()
    t = re.sub(r'(?<=\d)([+-])(\d)$', r'e\1\2', t)
    t = re.sub(r'(?<=\d)([+-])(\d{2,})$', r'e\1\2', t)
    try:
        return float(t)
    except ValueError:
        try:
            return int(t)
        except ValueError:
            return t

def legendre_coefficients_processing(data):    
    energy_coeff_dict = {}
    i = 0
    while i < len(data) - 1:
        row = data[i]
        if row[1] > 1 and row[0] == 0.0:
            current_energy = row[1] / 1e6
            coeffs = []
            i += 1
            while i < len(data):
                next_row = data[i]
                if next_row[0] == 0.0:
                    break
                clean_coeffs = [val for val in next_row if isinstance(val, float)]
                coeffs.extend(clean_coeffs)
                i += 1
            energy_coeff_dict[current_energy] = coeffs
        else:
            i += 1
    return energy_coeff_dict

def legendre_function_endf_format(data_dict):
        mu = sp.Symbol('mu')
        result = {}
        for E, coeffs in data_dict.items():
            NL = len(coeffs)
            expr = sp.Rational(1, 2) + sum(
                ((2*l + 1)/2) * coeffs[l-1] * sp.legendre(l, mu)
                for l in range(1, NL+1)
            )
            result[E] = expr
        return result

def angular_distributions(data_dict):
    result = {}
    exclude_values = {328.0, 409.0, 6.0, 50.0, 51.0}

    for E, values in data_dict.items():
        if not values:
            mu = sp.Symbol('mu')
            result[E] = 0 * mu
            continue
        
        filtered_values = [v for v in values if v not in exclude_values]
        angles = []
        ddx = []
        for i in range(0, len(filtered_values), 2):
            angles.append(filtered_values[i])
            ddx.append(filtered_values[i + 1])

        coef = np.polyfit(angles, ddx, 1)
        mu = sp.Symbol('mu')
        expr = coef[0] * mu + coef[1]

        result[E] = expr

    return result

def supplement_ground_excite(ground, excite):
    from sympy import symbols, simplify
    mu = symbols('mu')

    ground_keys = sorted(ground.keys())
    excite_keys = sorted(excite.keys())

    missing_keys = [k for k in ground_keys if k not in excite_keys]
    missing_key  = [l for l in excite_keys if l not in ground_keys]

    for key in missing_keys:
        excite[key] = 0
    
    for key in missing_key:
        lower_keys = [k for k in ground_keys if k < key]
        upper_keys = [k for k in ground_keys if k > key]

        if not lower_keys or not upper_keys:
            continue  

        lower = max(lower_keys)
        upper = min(upper_keys)

        expr_lower = ground[lower]
        expr_upper = ground[upper]

        ratio = (key - lower) / (upper - lower)
        interpolated_expr = simplify((1 - ratio) * expr_lower + ratio * expr_upper)

        ground[key] = interpolated_expr

    return ground, excite

def insert_data(E_incident, ddx_dict):
    import sympy as sp
    mu = sp.Symbol('mu')

    def scaling_DDX(E, E1, E2, ddx_dict):
        f1 = ddx_dict[E1]
        f2 = ddx_dict[E2]

        coeffs1 = sp.Poly(f1, mu).all_coeffs()
        coeffs2 = sp.Poly(f2, mu).all_coeffs()

        max_len = max(len(coeffs1), len(coeffs2))
        coeffs1 = [0] * (max_len - len(coeffs1)) + coeffs1
        coeffs2 = [0] * (max_len - len(coeffs2)) + coeffs2

        interp_coeffs = [
            c1 + (c2 - c1) * (E - E1) / (E2 - E1)
            for c1, c2 in zip(coeffs1, coeffs2)
        ]

        degree = max_len - 1
        f_interp = sum(c * mu**(degree - i) for i, c in enumerate(interp_coeffs))
        return sp.simplify(f_interp)

    import bisect

    E_ddx = sorted(ddx_dict.keys())
    inserted_dict = {}

    for e in tqdm(E_incident, desc="Interpolating DDX", unit="pt"):
        if e in E_ddx:
            inserted_dict[e] = ddx_dict[e]
        else:
            idx = bisect.bisect_left(E_ddx, e)
            if idx == 0 or idx == len(E_ddx):
                continue  
            e_lower = E_ddx[idx - 1]
            e_upper = E_ddx[idx]
            inserted_dict[e] = scaling_DDX(e, e_lower, e_upper, ddx_dict)  

    print(inserted_dict.keys())
    return inserted_dict
