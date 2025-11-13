
"""
transform the angle from cm frame to laboratory frame and each DDX value
・Reactions： p(7Li, n)7Be,  p(9Be, n)9B

Determination:
・γ value
・maximum angle
-- cm_to_ex_frame_angle()

・Neutrons' maximum neutron energy emission
-- energy_range_determination()

・DDX calculation for each energy and degree
-- cm_to_ex_frame_cross_section()

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from degree_frame import cm_to_ex_theta

def import_material_params(Material_params):
    # parameter import
    Ma = Material_params["Ma"]
    MA = Material_params["MA"]
    Mb = Material_params["Mb"]
    MB = Material_params["MB"]
    Q_ground  = Material_params["Q_ground"]
    Q_excite = Material_params["Q_excite"]
    if Q_excite is None:
        Q_excite = 0.0


    return Ma, MA, Mb, MB, Q_ground, Q_excite 

# γ value determination
def gamma_factor(Ma, MA, Mb, MB, Q, E, material):
    """
    a: incident ion (Li or Be)
    A: Target particle (Proton)
    b: Neutron
    B: Associate Particle (Be)
    E: Energy bins (MeV/n)
    """
    # multiply the mass number
    if material == "Li":
        mass_coeff = 7
    elif material == "Be":
        mass_coeff = 9

    # Energy convertion
    E_total  = mass_coeff*E
    velocity = np.sqrt((2*E_total)/Ma)
    reduced_mass = (Ma*MA)/(Ma+MA)
    E_bar    = reduced_mass* (velocity**2) / 2

    gamma = np.sqrt((Ma*Mb/(MA*MB))*((MB+Mb)/(MA+Ma))*(E_bar/(E_bar+Q)))

    return gamma

# Determination of the maximum emission angle of neutrons.
# It was determined in case of the maximum incident energy

def maximum_angle(Ma, MA, Mb, MB, Q, E_max, material):
    gamma = gamma_factor(Ma, MA, Mb, MB, Q, E_max, material)

    maximum_angle = np.arcsin((1/gamma)) 

    # rad to degree
    maximum_angle_deg = np.degrees(maximum_angle)

    return maximum_angle_deg

def particles_energy(theta, theta_L_deg, emission_energies, emission_frag_data, E_threshold = 1.5):
    """
    loading the energy of emitted neutrons for the degree.
    Their energy distribution was made based on the degree list in laboratory frame, 
    which is divided by {degree_step} in the range of 0-180 degrees.
    The processing steps are:
    1. get the closest degree from the degree list
    2. get the energy value of scattered neutrons

    Parameters
    ・theta: degree

    ・theta_L_deg: the degree list in laboratory frame, made of a degree lsit based on cm frame
      divided by {degree_step} in the range of 0-180 degrees.

    ・emission_energies: a list of scattered energy for theta_L_deg

    ・emission_frag_data:frag_data の書き込み用に設定された，中性子散乱エネルギーの元リスト

    """

    arr_theta = np.array(theta_L_deg) 
    arr_emission_energy = np.array(emission_energies)
    arr_emission_frag_data = np.array(emission_frag_data)


    epsilon = 0.1     # tolerance

    matching_indices = np.where(np.abs(arr_theta - theta) < epsilon)[0]

    if len(matching_indices) == 0:
        return []

    energy_candidates = arr_emission_energy[matching_indices]
    high_energies = energy_candidates[energy_candidates > E_threshold]
    low_energies  = energy_candidates[energy_candidates < E_threshold]

    E_high_avg = np.mean(high_energies) if len(high_energies) > 0 else None
    E_low_avg = np.mean(low_energies) if len(low_energies) > 0 else None
    
    corresponding_energy = [E_high_avg, E_low_avg]
    corresponding_energy_frag_data = []

    for e in corresponding_energy:
        if e is None or np.isnan(e):
            continue
        else:
            idx = np.argmin(np.abs(arr_emission_frag_data - e))
            if np.abs(arr_emission_frag_data[idx] - e) < epsilon:
                corresponding_energy_frag_data.append(arr_emission_frag_data[idx])

    return corresponding_energy_frag_data

# values in frag data are based on this energy.
def energy_range_determination(Ma, MA, Mb, MB, Q, theta, E, material):
    if material == "Li":
        mass_coeff = 7
    elif material == "Be":
        mass_coeff = 9

    E_total = mass_coeff * E
    
    theta_rad = np.radians(theta)

    def emission_energy(E_val):
        term1 = ((Ma * Mb) / (MA + MB)**2 + (MA * MB) / ((Ma + MA) * (Mb + MB))) * E_val
        term2 = (MB / (MB + Mb)) * Q
        term3 = 2 * np.sqrt((Ma * MA * Mb * MB / ((Ma + MA) * (Mb + MB)**3)) * (E_val**2 + ((Ma + MA) / MA) * Q * E_val)) * np.cos(theta_rad)
        return term1 + term2 + term3
    
    En = emission_energy(E_total)
    
    return En
