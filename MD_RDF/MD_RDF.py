import math
import os
from typing import IO, NoReturn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Program user options:
N_bins = 100 # No. of bins in RDF using pair particle histogram definition.
mode = 5 # Specify system: ideal gas (0), water O-O (1), water O-H (2), set1 (3), set2 (4), set3 (5)


def main() -> NoReturn:
    get_data()
    init_RDF()
    calc_RDF()
    normalise_RDF()
    output_RDF()
    viz_RDF()


def get_data() -> NoReturn:
    '''Reads an input file. Extracts MD parameters and trajectory data.'''
    if mode == 0:
        file_name = r"\ideal\ideal.xyz"
    elif mode == 1 or mode == 2:
        file_name = r"\water\water.xyz"
    elif mode == 3:
        file_name = r"\set1\set1.xyz"
    elif mode == 4:
        file_name = r"\set2\set2.xyz"
    elif mode == 5:
        file_name = r"\set3\set3.xyz"
    script_path = os.path.realpath(os.path.dirname(__file__))
    file_path = script_path + file_name
    with open(file_path) as input_file:
        get_params(input_file)
        get_trajectory(input_file)


def init_RDF() -> NoReturn:
    '''Creates a 2D array of zeroes with N_configs rows and  N_bins columns.
    Creates a 1D array of N_bins equidistant values between zero and max_box_length
    Adds bin_width and N_pairs keys to params, specifying histogram bin width and
    # particle pairs'''
    global RDF_r, RDF_f
    RDF_f = np.zeros((params["N_configs"], N_bins)) # Initialise RDF function values as a 2D array of zeros. 
    # Columns represent the bins (approx r value) that the RDF function values correspond to.
    # Rows represent the configurations that the RDF has been calculated from.
    max_box_length = ((params["xmax"] - params["xmin"]) ** 2
                    + (params["ymax"] - params["ymin"]) ** 2
                    + (params["zmax"] - params["zmin"]) ** 2) ** 0.5  # Q? What should the max box length be??
    RDF_r = np.linspace(0, max_box_length, N_bins) # 1D array of minimum distances (r) in each bin.
    params["bin_width"] = max_box_length / N_bins
    params["box_vol"] = (params["xmax"] - params["xmin"]) * (params["ymax"]
                    - params["ymin"]) * (params["zmax"] - params["zmin"])


def calc_RDF() -> NoReturn:
    '''Calculates RDF as a 1D array from an MD trajectory'''
    global RDF_f
    for config_index in range(params["N_configs"]): # Config_index ranges from 0 to no. configs in trajectory
        config = trajectory[config_index]
        RDF_config = calc_RDF_config(config) # Calculate RDF of each configuration in MD trajectory
        RDF_f[config_index] = RDF_config # Stores RDF for the configuration
    RDF_f = np.mean(RDF_f, axis = 0) # Obtain RDF as average over all configurations
    print(RDF_f)


def calc_RDF_config(config: np.ndarray) -> np.ndarray:
    '''Computes RDF as a 1D array for one configuration of particle positions.'''
    global params
    positions = get_positions(config)
    RDF_config = np.zeros((N_bins))
    N_pairs = 0 # Counter for no. of atom pairs included in pair list.
    # Loop over all particle pairs to calculate pair list and add each pair to appropriate histogram bin.
    for i in range(params["N"] - 2):  # Note: Counting particles from zero means final particle is (N-1)th particle
        for j in range(i + 1, params["N"] - 1):
            if mode == 1 and (positions[i, 0] != 1 or positions[j, 0] != 1):
                continue # For H2O O-O RDF, omit pairs including H atoms.
            if mode == 2 and (positions[i, 0] == positions[j, 0]):
                continue # For H2O O-H RDF, omit homonuclear pairs.
            # Calculate distances between atoms for each coordinate
            dx = positions[j, 1] - positions[i, 1]
            dy = positions[j, 2] - positions[i, 2]
            dz = positions[j, 3] - positions[i, 3]
            if mode in [1, 2, 3, 4, 5]: # Account for periodic boundary conditions via minimum image convention
                dx = dx - (params["xmax"] - params["xmin"]) * int(dx / ((params["xmax"] - params["xmin"])))
                dy = dy - (params["ymax"] - params["ymin"]) * int(dx / ((params["ymax"] - params["ymin"])))
                dz = dz - (params["zmax"] - params["zmin"]) * int(dx / ((params["zmax"] - params["zmin"])))
            r = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5   # Calculate pair distance
            bin_index = int(r / params["bin_width"]) # Specify which bin this distance belongs to.
            if bin_index < N_bins:
                RDF_config[bin_index] += 2 # Increment frequency by 2 for appropriate bin.
                N_pairs += 1
    params["N_pairs"] = N_pairs
    return RDF_config


def normalise_RDF() -> NoReturn:
    global RDF_f, RDF_r
    RDF_f = np.delete(RDF_f, 0) # Delete first bin to avoid division by zero
    RDF_r = np.delete(RDF_r, 0) # Delete first bin to avoid division by zero
    RDF_f = RDF_f / params["N_pairs"] # Divide by number of atom pairs
    shell_vol = 4 / 3 * math.pi * ((RDF_r ** 3) - (RDF_r + params["bin_width"] ** 3)) 
    RDF_f = RDF_f / shell_vol # Divide by shell vol for each bin
    RDF_f = RDF_f / (params["N"] / params["box_vol"]) # Divide by particle density
    print(RDF_f)


def output_RDF() -> NoReturn:
    RDF_df = pd.DataFrame(data = {"RDF_r": RDF_r, "RDF_f": RDF_f})
    if mode == 0:
        file_name = "\ideal\output.csv"
    elif mode == 1:
        file_name = "\water\output_OO.csv"
    elif mode == 2:
        file_name = "\water\output_OH.csv"
    elif mode == 3:
        file_name = "\set1\output.csv"
    elif mode == 4:
        file_name = "\set2\output.csv"
    elif mode == 5:
        file_name = "\set3\output.csv"
    script_path = os.path.realpath(os.path.dirname(__file__))
    file_path = script_path + file_name
    RDF_df.to_csv(file_path)
    print("Output complete")


def viz_RDF() -> NoReturn:
    plt.plot(RDF_r, RDF_f)
    plt.xlabel("Distance")
    plt.ylabel("RDF")
    plt.show()
    if mode == 0:
        file_name = "\ideal\graph.png"
    elif mode == 1:
        file_name = "\water\graph_OO.png"
    elif mode == 2:
        file_name = "\water\graph_OH.png"
    elif mode == 3:
        file_name = "\set1\graph.png"
    elif mode == 4:
        file_name = "\set2\graph.png"
    elif mode == 5:
        file_name = "\set3\graph.png"
    script_path = os.path.realpath(os.path.dirname(__file__))
    file_path = script_path + file_name
    plt.savefig(file_path)
    print("Visualisation complete")


def get_params(input_file: 'IO.TextIOWrapper') -> NoReturn:
    '''Reads an input file and extracts the MD simulation 
    parameters as a dictionary.'''
    global params
    params = {"N": None, "xmin": None, "xmax": None, "ymin": None,
            "ymax": None, "zmin": None, "zmax": None}
    flag_number = 0
    flag_box = 0
    for line in input_file:
        if flag_number == 0:
            if "ITEM: NUMBER OF ATOMS" in line:
                flag_number = 1
        elif flag_number == 1: # Reading N line
            line_split = line.split()
            params["N"] = int(line_split[0])
            flag_number = 0 
        if flag_box == 0:
            if "ITEM: BOX BOUNDS" in line:
                flag_box = 1 
        elif flag_box == 1: # Reading x bounds
            line_split = line.split()
            params["xmin"] = float(line_split[0])
            params["xmax"] = float(line_split[1])
            flag_box = 2
        elif flag_box == 2: # Reading y bounds
            line_split = line.split()
            params["ymin"] = float(line_split[0])
            params["ymax"] = float(line_split[1])
            flag_box = 3
        elif flag_box == 3: # Reading z bounds
            line_split = line.split()
            params["zmin"] = float(line_split[0])
            params["zmax"] = float(line_split[1])
            flag_box = 0
        if is_dict_full(params) == True:
            break
    print("Parameters:")
    print(params)


def get_trajectory(input_file: 'IO.TextIOWrapper') -> NoReturn:
    '''Reads an input file and extracts an MD trajectory as a list of 
    particle configurations. Adds # configs to params.'''
    global trajectory
    trajectory = []
    atom_count = None
    for line in input_file:
        if atom_count == None: # Seeking new configuration
            if "ITEM: ATOMS" in line: # Start of configuration data
                atom_count = 0
                config = np.empty((params["N"], 7))
        elif atom_count == params["N"]: # End of configuration data
            trajectory.append(config)
            atom_count = None
        else: # Reading configuration data
            line_split = [float(i) for i in line.split()]
            config[atom_count] = line_split
            atom_count += 1
    trajectory.append(config) # Loop ends before last config is added
    params["N_configs"] = len(trajectory)
    print(str(len(trajectory)) + " configurations read.")


def is_dict_full(dict: dict) -> bool:
    '''Returns true if no keys of a dictionary have None value. 
    Returns false otherwise'''
    is_full = True
    for i in dict.values():
        if i == None:
            is_full = False
    return is_full


def get_positions(config: np.ndarray) -> np.ndarray:
    '''Returns a 2D array of particle types/positions for a given particle configuration.
    Columns 1, 2, 3 and 4 represent atom type, x, y and z respectively (all Cartesian coordinates).'''
    positions = config[:, (2,4,5,6)] # Columns 2, 4, 5, 6 represent atom type, scaled x, scaled y, scaled z respectively.
    # Convert scaled coordinate to Cartesian coordinate.
    positions[:, 1] = positions[:, 1] * (params["xmax"] - params["xmin"]) 
    positions[:, 2] = positions[:, 2] * (params["ymax"] - params["ymin"])
    positions[:, 3] = positions[:, 3] * (params["zmax"] - params["zmin"])
    return positions 


if __name__ == "__main__":
    main()