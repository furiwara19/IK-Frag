# IK-Frag: Nuclear Data Generator for Inverse Kinematic Reactions in PHITS

## Overview
**IK-Frag** is a Python-based toolkit designed to generate **Frag data** for *inverse kinematic reactions* in PHITS.  
It aims to contribute to the future development of neutron source systems utilizing inverse kinematics by automating complex data preparation processes.

## Features
- **Automated nuclear data preparation**  
  IK-Frag automates the generation and formatting of nuclear data required for PHITS simulations of inverse kinematic reactions, which typically involve complex physics-based processing.

- **Flexible input configuration**  
  Users can define initial parameters for Frag data generation, such as:
  - Energy and angular bin settings  
  - Selected nuclear data libraries (e.g., JENDL, TENDL, ENDF)  
  - Projectile nucleus (e.g., Li or Be)

- **Python-based data processing**  
  Data acquisition, modification, and documentation are handled using Python modules such as **NumPy** and **SciPy**, ensuring accuracy and reproducibility.

- **Error reduction through automation**  
  By automating data handling and file writing based on the PHITS manual, IK-Frag minimizes human intervention and reduces the potential for manual errors.

## Installation
To install IK-Frag in your computer, please do this command in your cmd:

```bash
git clone https://github.com/furiwara19/IK-Frag.git
```

## Usage
To generate Frag data using **IK-Frag**, follow these steps:

1. **Launch the program**
```bash
cd <your location to IK-Frag folder>
python Ik_Frag.py
```
2. **Set parameters**

We need to define parameters of nuclear data to process, as listed below:
- projectiles nucleus: we focuses on Lithium and Berylium
- energy/degree bins: More precise data can be generated as detailed these parameters. Interpolation has been performed using SciPy module.
- Energy threshold of projectile: We can set the energy threshold for data processing to save the computational time while maintaining sufficient performance of generated Frag data.
Users can easily determined these parameters based on items in the GUI as follows,

![GUI overview](docs/IK-Frag_Discription_1.png)

![GUI overview](docs/IK-Frag_Discription_2.png)

By entering all parameters, every process is hundled automatically.

As for the nuclear data, users can extract it from IK-Frag/data_files folder.

## Dependencies
- Python 3.12 (Authors' environment) or higher
- Numpy
- Scipy
- Matplotlib (for visualization, optional)

## License
This project is licensed under the MIT License.

## Author
- Yu Fujiwara, Graduate School of Engineering, Osaka University, fjwryu.1120@gmail.com
- Toshiro Sakabe, Brookhaven National Laboratory, sakabe.toshiro.f21@kyoto-u.jp
- Masahiro Okamura, Brookhaven National Laboratory, okamura@bnl.gov
