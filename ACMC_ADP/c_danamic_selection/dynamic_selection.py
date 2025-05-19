

import numpy as np


def uncertainty_selection_optimized(predict_labels, data, neighbors, result_dict, nearest_neighbors, max_un_ins_num):
    predict_labels = np.array(predict_labels)


    flattened_neighbors_set = set([item for sublist in neighbors for item in sublist])  

    uncertainty_vec = []

    for i in range(len(data)):
        if i not in flattened_neighbors_set:
          
            neighbor_indices = nearest_neighbors[i]
            denominator = len(neighbor_indices)

            if denominator == 0:
                continue  

            neighbor_labels = predict_labels[neighbor_indices]  
            label_counts = np.bincount(neighbor_labels, minlength=len(result_dict))

            proportions = label_counts / denominator
            non_zero_proportions = proportions[proportions > 0] 
            entropy = -np.sum(non_zero_proportions * np.log2(non_zero_proportions))

            uncertainty_vec.append([i, entropy])

    if uncertainty_vec:
        uncertainty_vec = np.array(uncertainty_vec)
        sorted_indices = np.argsort(uncertainty_vec[:, 1])[::-1]
        top_m_points = uncertainty_vec[sorted_indices[:max_un_ins_num], 0].astype(int) 
    else:
        top_m_points = []
    # print(f'uncertainty_selection_optimized finish')

    return top_m_points








