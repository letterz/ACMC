import math
import networkx as nx
import numpy as np

from scipy.spatial.distance import euclidean,pdist
import numpy as np
from scipy.spatial.distance import cdist

from scipy.spatial import KDTree


def dc_cal_optimized(alpha, data):
    # 计算所有点对之间的欧几里得距离
    distances = pdist(data)  # 直接计算距离，返回一个一维数组，长度为 n(n-1)/2

    # 计算 dc 所对应的索引
    n = len(data)
    a = math.floor(0.5 * n * (n - 1) * alpha + 0.5)

    # 使用 np.partition 找到第 a 小的距离
    dc = np.partition(distances, a - 1)[a - 1]

    # print('dc_cal_optimized finished')
    return dc


def local_density_cal_optimized(nearest_neighbors, data, dc):
    # 预先计算所有点之间的欧几里得距离矩阵
    distances = cdist(data, data)  # 距离矩阵，避免重复计算

    # 计算局部密度
    density_vec = np.zeros(len(data))  # 初始化密度向量为零

    for i in range(len(data)):
        neighbors = nearest_neighbors[i]  # 取出第 i 个点的邻居
        neighbor_distances = distances[i, neighbors]  # 获取所有邻居的距离
        density_vec[i] = np.sum(np.exp(-(neighbor_distances / dc) ** 2))  # 矢量化计算密度

    # print('local_density_cal_optimized finished')
    return density_vec,distances

def nearest_higher_vec_optimized(density_vec, data,distances_matrix):
    n = len(data)
    distances_vec = []

    for i in range(n):
        # 找出密度比当前点大的所有点的索引
        higher_density_indices = np.where(density_vec > density_vec[i])[0]

        if len(higher_density_indices) == 0:
            # 如果没有密度比当前点大的点，最大距离作为默认值
            max_distance = np.max(distances_matrix[i, :])
            distances_vec.append([i, -1, max_distance])
        else:
            # 找出距离最近的密度更高的点
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
    # 1. 使用数据构建 KDTree
    tree = KDTree(data)  # 构建 KD-Tree

    # 2. 遍历每个数据点，查询其最近邻
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
    data += np.random.rand(*data.shape) * 0.000001  # 直接在data上操作
    # print('data preprocess finished')
    return data




