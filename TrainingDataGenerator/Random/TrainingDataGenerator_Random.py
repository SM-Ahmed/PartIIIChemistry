import numpy as np
import math
import random
import os
import math
from typing import List, NoReturn

n_atoms = 4 # number of atoms in system
n_training = 20 # Number of training set data points.
n_features = 3*n_atoms + 6 # Number of training set features.
a_max = 2 # Maximum cell length 

def generate_data() -> np.array:
    "Produce training set coords data"
    pos = generate_pos(n_atoms)
    cell = 0
    while cell == 0:
        cell = generate_cell()
    data = np.concatenate((pos, cell), axis = 0)
    return data

def output_GMIN(data: np.array, file_pos: int) -> NoReturn:  # Note: Output functions are redundant. Should merge.
    "Output data for a GMIN input coords file."
    script_path = os.path.realpath(os.path.dirname(__file__))
    file_name = "\GMIN_coords\coords" + str(file_pos) + ".txt"
    file_path = script_path + file_name
    np.savetxt(file_path, data)

def output_BayesOpt(data: np.array) -> NoReturn:
    "Output data for a BayesOpt input training file."
    script_path = os.path.realpath(os.path.dirname(__file__))
    file_name = r"\BayesOpt_training" + r"\training.txt"
    file_path = script_path + file_name
    np.savetxt(file_path, data)

def generate_cell() -> List:
    "Generate random cell parameters. Returns zero if cell volume is negative."
    a = random.random() * a_max
    b = random.random() * a_max
    c = random.random() * a_max
    alpha = random.random() * 2 * math.pi
    beta = random.random() * 2 * math.pi
    gamma = random.random() * 2 * math.pi
    V = calc_cell_vol(a, b, c, alpha, beta, gamma)
    if np.isreal(V) == False:
        return 0
    cell_list = [[a, b, c], [alpha, beta, gamma]]
    for row in cell_list:
        for i in range(len(row)):
            row[i] = truncate(row[i])
    return cell_list

def calc_cell_vol(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    "Calculates cell volume from cell parameters"
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    P = (1- cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 + cos_alpha * cos_beta * cos_gamma) ** 0.5
    V = a * b * c * P
    return V

def generate_pos(N: int) -> List:
    "Generate random fractional coordinates for N atoms"
    pos_list = []
    for i in range(N):
        rand_pos = []
        for j in range(3):
            rand_value = truncate(random.random())
            rand_pos.append(rand_value)
        pos_list.append(rand_pos)
    return pos_list

def truncate(num, dp = 1): # Adapted from code on StackExchange.
    """Returns a value truncated to a specific number of decimal places."""
    if not isinstance(dp, int):
        raise TypeError("decimal places must be an integer.")
    elif dp < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif dp == 0:
        return math.trunc(num)
    factor = 10.0 ** dp
    return math.trunc(num * factor) / factor

def main():
    training = np.empty([n_training, n_features])
    for i in range(n_training):    
        data = generate_data()
        data_as_row = data.flatten()
        training[i] = data_as_row
        output_GMIN(data, i + 1)
    output_BayesOpt(training)


if __name__ == "__main__":
    main()