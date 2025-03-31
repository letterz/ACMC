
from heapq import nsmallest
from operator import itemgetter
from scipy.spatial.distance import euclidean

def construct_query_list_for_max_uncertain_xi_list(max_uncertain_xi_list,  N,data):
    """
    :param max_uncertain_xi_list:
    :param dist_matrix:
    :param n:
    :param N:
    :return: 最大不确定度列表里每个点的查询列表字典
    """
    query_list_dict={}
    for xi in max_uncertain_xi_list:
        #为最大不确定度列表里的每个点构建查询队列
        xi_query_list=construct_query_list_Q_for_xi(xi, N,data)
        query_list_dict[xi]=xi_query_list
    # print(f'construct_query_list_for_max_uncertain_xi_list finished')
    return query_list_dict



def construct_query_list_Q_for_xi(max_uncertain_xi, neighborhood_N, data):
    """
    :param max_uncertain_xi: 在最大不确定度列表里的实例x*
    :param neighborhood_N: 每个类别的邻域
    :param data:
数据集
    :return: 查询列表 Q: [(xi, i_label), (xj, j_label)]
    """
    closest_x_list = []  # 用数组存储 (closest_instance, label, closest_dis)
    for label, neighbors in enumerate(neighborhood_N):
        closest_dis = float('inf')
        closest_instance = -1

        for instance in neighbors:
            if instance != max_uncertain_xi:
                temp_dis = euclidean(data[max_uncertain_xi], data[instance])
                if temp_dis < closest_dis:
                    closest_dis = temp_dis
                    closest_instance = instance

        if closest_instance != -1:
            closest_x_list.append([closest_instance, label, closest_dis])

    # 按照距离进行排序
    closest_x_list.sort(key=lambda x: x[2])
    # 返回去掉距离的结果
    closest_x = [(item[0], item[1]) for item in closest_x_list]
    return closest_x