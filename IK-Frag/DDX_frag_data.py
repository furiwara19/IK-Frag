"""
Rewrites the laboratory-frame DDX values to minimize data size when writing to the frag data file.
Specifically, when generating Frag data in PHITS, consecutive zeros can be represented as `0, -(N-1)` 
depending on the number `N` of consecutive zeros. 
For each incident energy of ⁷Li, a dictionary is prepared that stores an (n, m) matrix, 
where the vertical axis (rows) represents the scattered neutron energy (n) 
and the horizontal axis (columns) represents the scattering angle (m).
"""

def compress_zeros_frag_style(row):
    """
    Convert a 1D array into frag-data format (zero compression: always expressed as 0, -N pairs).
    """
    result = []
    i = 0
    n = len(row)

    while i < n:
        # Detect consecutive zeros
        zero_count = 0
        while i < n and row[i] == 0:
            zero_count += 1
            i += 1
        if zero_count == 1:
            result.append(0)
        elif zero_count > 1:
            result.extend([0, -(zero_count - 1)])

        # Handle non-zero segments
        while i < n and row[i] != 0:
            result.append(row[i])
            i += 1

    return result


def transform_ddx_frag_style_robust(ddx):
    """
    Convert each DDX_matrix corresponding to a given incident energy into frag-data format (with zero compression).
    
    Parameters:
        ddx : list of (n, m) matrices
            A list containing the DDX matrices for each incident energy.
    """
    transformed_list = []

    for matrix in ddx:  # For each incident energy's ndarray
        transformed_matrix = []
        for row in matrix:
            compressed_row = compress_zeros_frag_style(row)
            transformed_matrix.append(compressed_row)
        transformed_list.append(transformed_matrix)
    
    return transformed_list
