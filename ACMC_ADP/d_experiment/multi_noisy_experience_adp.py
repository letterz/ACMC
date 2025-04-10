import os.path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ACMC_ADP.a_initailization.initialization import initialization
from ACMC_ADP.b_static_selection.static_selection import (neighbors_labeling, descending_order, sliding_window
    )
from ACMC_ADP.c_danamic_selection.dynamic_selection import (uncertainty_selection_optimized,)
from ACMC_ADP.ensemble.a_pre_cluster import (pre_cluster_user_vote_thread)
from ACMC_ADP.ensemble.b_construct_query_list import construct_query_list_for_max_uncertain_xi_list
from ACMC_ADP.ensemble.c_iteration_stage_user_vote import iteration_stage_user_vote_thread
from memory_profiler import profile,memory_usage


# @profile
def experiemnt_ACMC_adp_thread(data, real_labels,alpha,l,theta,users_list,min_users_num, max_users_num,max_uncertain_xi_num,user_locks,isUpdate=True):
    ARI_record=[]
    ARI= adjusted_rand_score(real_labels, [0] * len(data))
    interaction=0
    iter = 0
    count = 0
    constraints_num = 0
    ARI_record.append([{"iter": iter, "interaction": count, "constraints_num":constraints_num, "ari": ARI}])
    dc,density_vec,distances_vec,center_vec,ascription_tree,nearest_neighbors, = initialization(alpha, data)
    center_vec_dec = descending_order(center_vec)
    P = sliding_window(l, center_vec_dec, theta)

    # 用户轮流判断
    neighbors, users_list, count,constraints_num = pre_cluster_user_vote_thread(P, users_list,
                                                               real_labels, min_users_num, max_users_num, count,
                                                               user_locks,constraints_num,isUpdate)
    # print("centroids:", neighbors)
    predict_labels, result_dict = neighbors_labeling(ascription_tree, neighbors)
    while (True) :
        top_m_un_points = uncertainty_selection_optimized(predict_labels, data, neighbors, result_dict,nearest_neighbors,max_uncertain_xi_num)

        if len(top_m_un_points) == 0:
            # print("查询完毕")
            break
        else:
            top_m_un_points = [int(item) for item in top_m_un_points]
        query_list_dict = construct_query_list_for_max_uncertain_xi_list(top_m_un_points, neighbors,data)
        #用户轮流判断
        neighbors, users_list, count, max_uncertain_xi_list_label_dict,constraints_num = iteration_stage_user_vote_thread(
            top_m_un_points, query_list_dict, users_list, real_labels, count, neighbors,min_users_num, max_users_num, user_locks,constraints_num,isUpdate)

        predict_labels, result_dict = neighbors_labeling(ascription_tree, neighbors)
        iter=iter+1
        ARI = adjusted_rand_score(real_labels, predict_labels)
        ARI_record.append([{"iter": iter, "interaction": count,"constraints_num":constraints_num, "ari": ARI}])

        if ARI==1:
            break
    return ARI_record

def result_to_csv(ARI_record, title,output_path):
    os.makedirs(output_path, exist_ok=True)
    # 将所有的交互到加入到里面去`
    record = []
    for i in range(len(ARI_record)):
        if i<len(ARI_record)-1:
            repeat=ARI_record[i+1][0]["interaction"]-ARI_record[i][0]["interaction"]
            for j in range(repeat):
                record.append(ARI_record[i][0]["ari"])
        else:
            record.append(ARI_record[i][0]["ari"])
    # 写入文档
    df = pd.DataFrame({
        'ARI': record,
    })
    fullpath=os.path.join(output_path,f'{title}.csv')
    df.to_csv(fullpath,mode='w')
def get_data_from_datasets(file_path):
    true_label = []
    data_matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行末的换行符然后以逗号分割数据
            line_data = line.strip().split(',')
            # 将除了最后一个数据之外的数据添加到列表
            row_data = np.array([float(x) for x in line_data[:-1]])
            data_matrix.append(row_data)
            true_label.append(int(float(line_data[-1])))
    # 转换成ndarray样式
    data_matrix = np.array((data_matrix))


    return data_matrix,true_label
if __name__ == '__main__':

    pass








