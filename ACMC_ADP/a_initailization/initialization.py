import math
import networkx as nx
import numpy as np

from scipy.spatial.distance import euclidean,pdist
import numpy as np
from scipy.spatial.distance import cdist

from scipy.spatial import KDTree


def dc_cal_optimized(alpha, data):
    
    distances = pdist(data)  
    n = len(data)
    a = math.floor(0.5 * n * (n - 1) * alpha + 0.5)

    dc = np.partition(distances, a - 1)[a - 1]

    # print('dc_cal_optimized finished')
    return dc


def local_density_cal_optimized(nearest_neighbors, data, dc):
    distances = cdist(data, data) 

    density_vec = np.zeros(len(data)) 

    for i in range(len(data)):
        neighbors = nearest_neighbors[i] 
        neighbor_distances = distances[i, neighbors] 
        density_vec[i] = np.sum(np.exp(-(neighbor_distances / dc) ** 2)) 

    # print('local_density_cal_optimized finished')
    return density_vec,distances

def nearest_higher_vec_optimized(density_vec, data,distances_matrix):
    n = len(data)
    distances_vec = []

    for i in range(n):
        higher_density_indices = np.where(density_vec > density_vec[i])[0]

        if len(higher_density_indices) == 0:
          
            max_distance = np.max(distances_matrix[i, :])
            distances_vec.append([i, -1, max_distance])
        else:
          
            min_distance_index = np.argmin(distances_matrix[i, higher_density_indices])
            nearest_higher_index = higher_density_indices[min_distance_index]
            min_distance = distances_matrix[i, nearest_higher_index]
            distances_vec.append([i, nearest_higher_index, min_distance])

    # print('nearest_higher_vec_optimized finished')
    return distances_vec


def ascription_tree_construction(distances_vec):
    ascription_tree=nx.DiGraph()
    ascription_tree.add_weighted_edges_from(distances_vec)
    ascription_tree.remove_node(-1)
    return ascription_tree


def center_probability_cal(density_vec,distances_vec):
    a=np.array(density_vec)
    b=np.array(distances_vec)[:,2]
    center_vec=[]
    max_a=np.max(a)
    max_b=np.max(b)
    for i in range(len(a)):
        center_vec.append([i,(a[i]*b[i])/(max_a*max_b)])
    # print(f'center_probability_cal finished')
    return center_vec

def nearest_neighbors_cal_kdtree(dc, data):

    tree = KDTree(data) 

    nearest_neighbors = [tree.query_ball_point(data[i], dc) for i in range(len(data))]

    # print('nearest_neighbors_cal_kdtree finished')
    return nearest_neighbors




def initialization(alpha,data):
    data = data_preprocess(data)
    # dist_matrix = cdist(data, data)
    dc=dc_cal_optimized(alpha, data)

    nearest_neighbors=nearest_neighbors_cal_kdtree(dc, data)

    density_vec,distance_matix=local_density_cal_optimized(nearest_neighbors, data,dc)

    distances_vec=nearest_higher_vec_optimized(density_vec, data,distance_matix)
    center_vec=center_probability_cal(density_vec,distances_vec)
    ascription_tree = ascription_tree_construction(distances_vec)
    return dc,density_vec,distances_vec,center_vec,ascription_tree,nearest_neighbors,

def data_preprocess(data):
    np.random.seed(0)
    data += np.random.rand(*data.shape) * 0.000001
    # print('data preprocess finished')
    return data




