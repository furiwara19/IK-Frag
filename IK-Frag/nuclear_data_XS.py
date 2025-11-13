import numpy as np

def get_XS(file_path, material, energy_threshold):
    """
    loading total cross-section from the selected nuclear data library

    Parameters:
        file_path (str): selected file path

    Returns:
        dict:
            {
                "ground": {energy (eV): cross_section (barn), ...},
                "excite": {energy (eV): cross_section (barn), ...}
            }
    """

    with open(file_path, errors='ignore') as f:
        lines = f.readlines()

    parts = [s.strip() for s in lines[0].split('/')]
    lib = parts[0]

    # marker set for material
    if material == "Li":
        marker_ground = "328 3 50"
        marker_excite = "328 3 51"
    else: #Be
        if "ENDF" in lib.upper():
            marker_ground = "425 3 50"
            marker_excite = "425 3 51" 
        else:
            marker_ground = "409 3 50"
            marker_excite = "409 3 51"        
    

    if "ENDF" in lib.upper() or "JENDL" in lib.upper():
        ground_data = find_target_data(lines, marker_ground, threshold=energy_threshold)
        excite_data = find_target_data(lines, marker_excite, threshold=energy_threshold)
        ground_dict = data_processing_to_dict(ground_data)
        excite_dict = data_processing_to_dict(excite_data)
    else:
        ground_data = find_target_data(lines, marker_ground, threshold=energy_threshold)
        ground_dict = data_processing_to_dict(ground_data)
        # TENDL doesn't have any data for excite reaction.
        excite_dict = {key: 0 for key in ground_dict.keys()}
        
        
    return {
        "ground": ground_dict,
        "excite": excite_dict
    }

def find_target_data(lines, marker, threshold):
    # reading nuclear data values
    target_data = []
    reading = False
    counter = 0
    for line in lines:
        if marker in line:
            counter += 1
            if counter < 4:
                continue
            elif counter == 4:
                reading = True
        else:
            if reading:
                break
            else:
                continue

        elements = line.split()
        converted_elements = []
        
        for idx, item in enumerate(elements):
            try:
                val = convert_to_float(item) 
            except ValueError:
                converted_elements = []
                break

            if idx % 2 == 0 and val > threshold:
                reading = False
                break

            converted_elements.append(val)

        if not reading:
            break

        if len(converted_elements) >= 6:
            target_data.append(converted_elements[:6])

    return target_data

def convert_to_float(value):
    if '-' in value:
        return float(value.replace('-', 'e-'))
    elif '+' in value:
        return float(value.replace('+', 'e+'))
    else:
        return float(value)
    
def data_processing_to_dict(target_data):
    result = {}
    for item in target_data:
        for i in range(0, 6, 2):
            energy = item[i]/1e6
            xs = item[i+1]
            result[energy] = xs
    return result


def interpolate_cross_section(XS_dict, energy_grid, method='linear'):    
    from scipy.interpolate import interp1d
    import numpy as np
    
    """
    interpolate XS values based on energy_grid

    Parameters:
        file_path (str): a file including nuclear data (.dat file)
        energy_grid (np.ndarray): Energy grid（unit: eV）
        method (str): 'log' or 'linear'

    Returns:
        dict: {
            "ground": np.ndarray,
            "excite": np.ndarray
        }
    """

    result = {}

    for state in ["ground", "excite"]:
        energy = np.array(list(XS_dict[state].keys()))
        xs = np.array(list(XS_dict[state].values()))

        if method == 'log':
            interp_func = interp1d(np.log10(energy), np.log10(xs), bounds_error=False, fill_value='extrapolate')
            log_interp_xs = interp_func(np.log10(energy_grid))
            interpolated = 10 ** log_interp_xs
        elif method == 'linear':
            interp_func = interp1d(energy, xs, bounds_error=False, fill_value='extrapolate')
            interpolated = interp_func(energy_grid)
        else:
            raise ValueError("method must be either 'log' or 'linear'")

        interpolated[interpolated < 0] = 0.0
        result[state] = interpolated
    
    return result