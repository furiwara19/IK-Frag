# メイン関数

"""
・Purpose
To enable the evaluation of the inverse kinematic reaction p(⁷Li, n)⁷Be in PHITS by generating the corresponding frag_data.

・Overview of auxiliary functions (see each file for details)
nuclear_data_XS.py: Retrieves the total collision cross section from each library.
nuclear_data_DDX.py: Retrieves the double differential cross section (DDX) data from each library.
neutron_energy_angle.py: Converts angles from the center-of-mass frame to the laboratory frame and determines the maximum energy of scattered neutrons.
DDX_data_processing.py: Derives the double differential cross section values.
DDX_frag_data.py: Rewrites the laboratory-frame DDX values to minimize data size when writing to the frag data file.

・Functions in this script
formatted_value(): Formats the retrieved list data for writing to frag_data (since the number of values per line is limited).

Procedure:
1. Obtain the cross-section values for both ground and excited state reactions, extrapolate them to match the energy bins, and sum them.
2. Determine the upper limit of the scattering angle for emitted neutrons.
3. Determine the maximum energy of emitted neutrons.
4. Register the double differential cross-section (DDX) data.
"""

# module import
from nuclear_data_XS import get_XS, interpolate_cross_section
from nuclear_data_DDX import get_DDX, supplement_ground_excite, insert_data
from neutron_energy_angle import maximum_angle, energy_range_determination, import_material_params
from DDX_data_processing import merge_ddx
from DDX_frag_data import compress_zeros_frag_style, transform_ddx_frag_style_robust
import numpy as np
from tqdm import tqdm

class frag_data_writing:
    def __init__(self):
        # material parameters
        self.Ma = None
        self.MA = None
        self.Mb = None
        self.MB = None
        self.Q_ground  = None
        self.Q_excite  = None
        
        # material cross-sections
        self.XS = None
        self.XS_ground = None
        self.XS_excite = None
        self.DDX = None
        self.DDX_ground = None
        self.DDX_excite = None

        # structured data (processed for frag data generation)
        self.incident_energy = None
        self.neutron_energy = None
        self.neutron_emission_degree = None
        self.DDX_matrix = None
        self.material_params = None

        # User defined
        self.nuclear_data = "JENDL"
        self.material   = "Li"   # Lithium or Berylium
        self.step_energy = 0.01  # energy step for frag data
        self.step_degree = 0.05  # degree step for frag data
        self.threshold_energy = 5e6  # threshold energy for frag data 
    
    def load_material(self):
        if self.nuclear_data == 'JENDL':
            reaction_params = {
                "Li": {
                    "Ma": 6.955734,
                    "MA": 0.991673,
                    "Mb": 1.0,
                    "MB": 6.956651,
                    "Q_ground": -1.644,
                    "Q_excite": -2.0733,
                },
                "Be": {
                    "Ma": 8.934764,  
                    "MA": 0.9991673,  
                    "Mb": 1.0,  
                    "MB": 8.935900,  
                    "Q_ground": -1.85
                }
            }
        elif self.nuclear_data == 'TENDL':
            reaction_params = {
                "Li": {
                    "Ma": 6.9541,
                    "MA": 0.9986234,
                    "Mb": 1.0,
                    "MB": 6.9545,
                    "Q_ground": -1.644,
                    "Q_excite": None
                },
                "Be": {
                    "Ma": 8.934761,   
                    "MA": 0.9991673,  
                    "Mb": 1.0,   
                    "MB": 8.935900,   
                    "Q_ground": -1.85,
                    "Q_excite": -3.4506
                }
            }

        else:
            reaction_params = {
                "Li": {
                    "Ma": 6.955734,
                    "MA": 0.9986234,
                    "Mb": 1.0,
                    "MB": 6.956651,
                    "Q_ground": -1.644,
                    "Q_excite": -2.0733
                },
                "Be": {
                    "Ma": 8.934761,   
                    "MA": 0.9991673,  
                    "Mb": 1.0,   
                    "MB": 8.935900,   
                    "Q_ground": -1.85,
                    "Q_excite": -3.450 
                }
            }

        key = self.material.capitalize() 
        if key not in reaction_params:
            raise ValueError(f"Unknown material: {self.material}")

        self.material_params = reaction_params[key]

        print(f"[INFO] Material parameters loaded for {self.material}")

    def load_XS(self, file_path):
        """Reading total cross section from selected nuclear data"""
        """ get XS data, energy_step, total-cross-section"""
        print("[INFO] Loading XS nuclear data...")
        result = get_XS(file_path, self.material, self.threshold_energy)
        ground_energies = np.array(list(result['ground'].keys())) 
        min_val = ground_energies.min()
        max_val = ground_energies.max()

        step=self.step_energy


        raw_grid = list(np.arange(
        np.floor((min_val + step) / step) * step,  
        max_val + step, 
        step
        ))

        truncated_grid = [np.floor(e * 100) / 100 for e in raw_grid] 

        energy_grid = [min_val] + truncated_grid
        
        self.incident_energy = energy_grid

        # linear interpolation
        # interpolated_log = interpolate_cross_section(file_path, energy_grid, method='log')
        interpolated_lin = interpolate_cross_section(result, energy_grid, method='linear')

        # sum up each XS values
        interpolated_lin['total'] = interpolated_lin["ground"] + interpolated_lin["excite"]

        self.XS = interpolated_lin["total"]
        self.XS_ground = interpolated_lin["ground"]
        self.XS_excite = interpolated_lin["excite"]
    
    def load_DDX(self, file_path):
        """Loading DDX values"""
        DDX_ground, DDX_excite = get_DDX(file_path, self.material, self.threshold_energy)
        print("[INFO] Loading DDX values from nuclear data...")
        ground, excite = supplement_ground_excite(DDX_ground, DDX_excite)
        
        self.DDX_ground = insert_data(self.incident_energy, ground)
        self.DDX_excite = insert_data(self.incident_energy, excite)
    
    def load_neutron_data(self): 
        """Loading neutron data of scattered neutrons and their emission angles"""
        maximum_incident_energy = max(self.incident_energy)
        Ma, MA, Mb, MB, Q_ground, Q_excite = import_material_params(self.material_params)
        maximum_neutron_degree  = maximum_angle(Ma, MA, Mb, MB, Q_ground, 
                                                maximum_incident_energy, self.material)

        step = self.step_degree
        self.neutron_emission_degree = np.arange(0, maximum_neutron_degree + step, step, dtype=np.float32)

        theta_min = 180
        theta_max = 0  

        En_min = energy_range_determination(Ma, MA, Mb, MB, Q_ground, theta_min, 
                                            maximum_incident_energy, self.material)
        En_max = energy_range_determination(Ma, MA, Mb, MB, Q_ground, theta_max, 
                                            maximum_incident_energy, self.material)

        self.neutron_energy = np.arange(En_min, En_max + step, step, dtype=np.float32)

    def DDX_arrangement(self):
        print("[INFO] Assigning DDX matrix...")
        ddx_matrix, _theta_maps, _energy_maps, _gammas = merge_ddx(
            self.material_params, self.incident_energy, self.neutron_emission_degree, 
            self.neutron_energy, self.material, self.DDX_ground, self.DDX_excite, 
            self.XS_ground, self.XS_excite, self.step_degree
            )
        # print(ddx_matrix)
        print(len(ddx_matrix))
        self.DDX_matrix = []
        for i in tqdm(range(len(ddx_matrix)), desc="generating DDX", unit="energy"):
            matrix = ddx_matrix[i]
            compressed = transform_ddx_frag_style_robust([matrix])[0]
            total_DDX = np.sum(matrix)
            self.DDX_matrix.append({
                'total': total_DDX,
                'compressed': compressed
            })

    def formatted_value(self, values):
        formatted_lines = []
        for i in range(0, len(values), 7):
            chunk = values[i:i+7]
            line = "  ".join(f"{v:g}" for v in chunk)
            formatted_lines.append(line + "\n")
        return formatted_lines

    def write_to_file(self, filename="frag_data.dat"):
        # --- Determination of the incidents and target based on material input ---
        key = self.material.capitalize()
        if key == "Li":
            beam, target = "7Li", "1H"     # Li beam × proton target
        elif key == "Be":
            beam, target = "9Be", "1H"     # Be beam × proton target
        else:
            raise ValueError(f"Unknown material: {self.material}")
        
        with open(filename, "w") as f:
            f.write(f"{beam}\n")
            f.write(f"{target}\n")

            f.write(f"{len(self.incident_energy) - 1}\n")
            f.writelines(self.formatted_value(self.incident_energy))  

            scaled_XS = [round(x * 1000, 3) for x in self.XS] # XS, unit= mb
            f.writelines(self.formatted_value(scaled_XS))

            f.write(f"{len(self.neutron_energy)}\n") # energy bins for scattering neutrons
            f.writelines(self.formatted_value(self.neutron_energy)) # neutron energy
            f.write(f"{-(len(self.neutron_emission_degree))}\n") # The number of emission energies
            f.writelines(self.formatted_value(self.neutron_emission_degree)) # corresponding degrees

            f.write("1\n")  
            f.write("neutron\n")  

            for j in range(len(self.DDX_matrix)): 
                f.write(f"{round(self.XS[j] * 1000, 3):.3f}\n") # DDX values, unit= mb
                for row in self.DDX_matrix[j]["compressed"]:
                    f.writelines(self.formatted_value(row))
                f.write("\n") 