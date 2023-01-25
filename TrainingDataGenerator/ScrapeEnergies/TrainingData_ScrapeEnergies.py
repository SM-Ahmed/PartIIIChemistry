import os
import numpy as np

N = 20 # No. of response data points
E = 0.1000000000E+21 # Energy value for poor data that returns empty markov files.
response = np.zeros(20)
print(response)

script_path = os.path.realpath(os.path.dirname(__file__))
folder_name = r"\FreezeCopies\train."
file_name = r"\markov"
is_good_data = False

for i in range(N):
    file_path = script_path + folder_name + str(i + 1) + file_name
    with open(file_path) as file:
        for line in file: # Only good data (i.e. markov file not empty) will run this code block
            energy = line.split()[1]
            response[i] = energy
            is_good_data = True 
            break # Only want first line in markov file
        if is_good_data == False: 
            response[i] = E
        else:
            is_good_data = False 

print(response)
np.savetxt("response", response)
print("Output complete")
