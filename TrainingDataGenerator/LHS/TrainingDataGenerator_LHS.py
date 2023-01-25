import math
import os
from typing import NoReturn
import numpy as np
from smt.sampling_methods import LHS

n_atoms = 4 # Number of atoms in simulated system
n_configs = 22 # Number of configurations to be generated
n_features = 3*n_atoms + 6 # Number of training set features.
a_max = 10 # Cell length upper bound
V_min = 0.25 # Cell volume lower bound.
is_orthorhombic = True # If true, fix cell angles at 90 deg.

def generate_cell() -> np.array:
    "Generate cell parameters via LHS"
    fract_coord_limits = [[0, 1] for i in range(3 * n_atoms)]
    if is_orthorhombic == False:
        angle_limits = [[0, 2 * math.pi] for i in range(3)]
    else:
        angle_limits = [[math.pi / 2, math.pi / 2] for i in range(3)]
    length_limits = [[0, a_max] for i in range(3)]
    limits = np.array(fract_coord_limits + angle_limits + length_limits)
    sampling = LHS(xlimits = limits)
    training_data = sampling(n_configs)
    return training_data

def remove_bad_rows(data: np.array) -> np.array:
    "Remove rows corresponding to data points with unphysical volumes."
    row_number = 0
    bad_rows = []
    for data_point in data:
        V = calc_cell_vol(data_point[-6:])
        if not ((np.isreal(V) == True) and (V > V_min)): # Vol of a good row is real and greater than V_min.
            bad_rows.append(row_number)
        row_number += 1
    data = np.delete(data, bad_rows, axis = 0)
    if is_orthorhombic == True:
        data = np.delete(data, (-4, -5, -6), axis = 1) # Remove cell angles
    return data

def analyse_training(data: np.array) -> NoReturn:
    "Print number of data points that the script generated, removed and outputted."
    n_training = np.shape(data)[0]
    n_bad_rows = n_configs - n_training
    print("Out of " + str(n_configs) + " data points generated,\n" 
        + str(n_bad_rows) + " data points were removed \n" 
        + "to make a training set of " + str(n_training) + " points.")


def output_BayesOpt(data: np.array) -> NoReturn:
    file_name = r"\BayesOpt_training" + r"\training"
    output_file(data, file_name)

def output_GMIN(data: np.array) -> NoReturn:
    if is_orthorhombic == False:
        n_rows = n_atoms + 2 # No. of rows in GMIN coords file
    else:
        n_rows = n_atoms + 1 # One fewer row because cell angles removed.
    file_pos = 1
    for data_point in data:
        coords = np.reshape(data_point, (n_rows, 3))
        file_name = "\GMIN_coords\coords" + str(file_pos)
        output_file(coords, file_name)
        file_pos += 1


def output_file(data: np.array, file_name: str) -> NoReturn:
    "Output file with specified data and file name."
    script_path = os.path.realpath(os.path.dirname(__file__))
    #GMIN: file_name = "\GMIN_coords\coords" + str(file_pos)
    file_path = script_path + file_name
    np.savetxt(file_path, data)


def calc_cell_vol(cell_params: list) -> float:
    "Calculates cell volume from cell parameters"
    cos_alpha = math.cos(cell_params[0])
    cos_beta = math.cos(cell_params[1])
    cos_gamma = math.cos(cell_params[2])
    a = cell_params[3]
    b = cell_params[4]
    c = cell_params[5]
    P = (1- cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 + cos_alpha * cos_beta * cos_gamma) ** 0.5
    V = a * b * c * P
    return V


def main():
    training = generate_cell()
    training = remove_bad_rows(training)
    analyse_training(training)
    output_BayesOpt(training)
    output_GMIN(training)

if __name__ == "__main__":
    main()