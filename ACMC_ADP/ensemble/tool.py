import numpy as np


def get_data_from_datasets(file_path):
    true_label = []
    data_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
           
            line_data = line.strip().split(',')
           
            row_data = np.array([float(x) for x in line_data[:-1]])
            data_matrix.append(row_data)
            true_label.append(int(float(line_data[-1])))
  
    data_matrix = np.array((data_matrix))

    return data_matrix,true_label
