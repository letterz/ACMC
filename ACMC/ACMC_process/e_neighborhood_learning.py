import copy
import csv
import heapq
import multiprocessing
import os
import traceback

import networkx as nx
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from memory_profiler import profile,memory_usage
import math

from ACMC.ACMC_process.a_DMCons import graph_initialization
from ACMC.ACMC_process.b_ranking_allocation import order_allocation
from ACMC.ACMC_process.c_neighborhood_initialization import connections_cal, interaction_process,neighborhood_initialization
from ACMC.ACMC_process.d_influence_model_propagation import influence_model_propagation

from ACMC.ACMC_process.ensemble.b_construct_query_list import construct_query_list_for_max_uncertain_xi_list
from ACMC.ACMC_process.ensemble.c_iteration_stage_user_vote import iteration_stage_user_vote_process
def k_nearest_neighbor_cal(data,k):

    neighbors = NearestNeighbors(n_neighbors=k).fit(data)
    k_nearest_neighbors = neighbors.kneighbors(data, return_distance=False)
    return k_nearest_neighbors


def uncertainty_oneNode(predict_labels, k_nearest_neighbor,k):
    dict={}
    for i in range(len(k_nearest_neighbor)):
        point=k_nearest_neighbor[i]
        if predict_labels[point] not in dict.keys():
            dict[predict_labels[point]]=[point]
        else:
            dict[predict_labels[point]].append(point)
    sum=0
    for m in dict.keys():
        proportion=len(dict[m])/k
        if proportion != 0:
            sum = sum + proportion * math.log2(proportion)
    sum = -sum
    if sum==-0.0:
        sum=0.0
    return sum

def uncertainty_cal(predict_labels,k_nearest_neighbors,candidates,k):
    uncertainty_dict=dict()
    for candidate in candidates:
        k_nearest_neighbor=k_nearest_neighbors[candidate]
        uncertainty=uncertainty_oneNode(predict_labels, k_nearest_neighbor,k)
        uncertainty_dict[candidate]=uncertainty
    return uncertainty_dict

def first_n_nodes_cal(my_dict,n):
    if n>len(my_dict):
        n=len(my_dict)
    sliced_list=[]

    heap = [(-value, key) for key, value in my_dict.items()]
    heapq.heapify(heap)
    count=0
    for _ in range(n):
        if heap:
            neg_value, key = heapq.heappop(heap)
            value = -neg_value
            if value==0.0:
                break
            sliced_list.append(key)
            del my_dict[key]
            count=count+1
        else:
            break
    remaining_count = n - count
    if remaining_count > 0:
        for i in range(remaining_count):
            first_key, first_value = next(iter(my_dict.items()))
            del my_dict[first_key]
            sliced_list.append(first_key)
            count = count + 1
    return sliced_list,my_dict

def result_to_csv_ACDM_thread(ARI_record, title,output_path):
    os.makedirs(output_path, exist_ok=True)
    # 打开文件进行写入
    fullpath = os.path.join(output_path, f'{title}_result.csv')
    with open(fullpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['interaction','constraints_num', 'ari'])
        # 遍历数据并提取 'interaction' 和 'ari'，然后写入文件
        for record_group in ARI_record:
            for record in record_group:
                interaction = record.get('interaction', None)
                constraints_num=record.get('constraints_num', None)
                ari = record.get('ari', None)
                writer.writerow([interaction,constraints_num, ari])
    return 0

def neighborhood_learning(skeleton, data, predict_labels, neighborhood, k_nearest_neighbors, count, order,real_labels, record,k,users_list,user_locks,min_users_num, max_users_num,max_uncertainty_num,constraints_num,pool,query_times_lock,constraints_num_lock,neighbourhoods_lock,label_dict_lock,rest_point_lock,manager,isUpdate):
    candidates = dict()
    for i in range(len(order)):
        candidates[order[i]] = 0
    flag = False
    iter=2
    while (True):
        candidates = uncertainty_cal(predict_labels, k_nearest_neighbors, candidates,k)
        sliced_list,candidates=first_n_nodes_cal(candidates, max_uncertainty_num)

        #ensemble代码
        query_list_dict = construct_query_list_for_max_uncertain_xi_list(sliced_list, neighborhood,data)
        count = count
        constraints_num = constraints_num
        neighborhood = neighborhood
        if pool._state == 'RUN':
            neighborhood, users_list, count, max_uncertain_xi_list_label_dict, constraints_num = iteration_stage_user_vote_process(
                sliced_list, query_list_dict, users_list, real_labels, count, neighborhood, min_users_num, max_users_num,
                user_locks, constraints_num,pool,query_times_lock,constraints_num_lock,neighbourhoods_lock,label_dict_lock,rest_point_lock,manager,isUpdate)
        else:
            raise RuntimeError("进程池在运行时被关闭")

        if candidates == dict():
            flag = True
        predict_labels=influence_model_propagation(skeleton, neighborhood)
        ari = adjusted_rand_score(real_labels, predict_labels)
        record.append([{"iter": iter, "interaction": count,"constraints_num":constraints_num, "ari": ari}])
        iter=iter+1
        if flag == True:
            break
        if ari ==1:
            break

    return record


def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec


def initialization_cut(skeleton,m,start_node):
    G=copy.deepcopy(skeleton)
    traversed_nodes = [start_node]
    candidate_edge = []
    for edge in G.edges(start_node, data=True):
        heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    while candidate_edge:
        max_edge = heapq.heappop(candidate_edge)
        G.remove_edge(max_edge[1],max_edge[2])
        if len(traversed_nodes)==m:
            break
        weight, current_node, new_node = -max_edge[0], max_edge[1], max_edge[2]
        traversed_nodes.append(new_node)
        for edge in G.edges(new_node, data=True):
            if edge[1] != current_node:
                heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    clusters = []
    S = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels


# @profile
def ACMC_process(data, real_labels,k,users_list,min_users_num, max_users_num,max_uncertainty_num,isUpdate=True):

    k_nearest_neighbors = k_nearest_neighbor_cal(data, k)
    skeleton, representative = graph_initialization(data)
    record = [[{"iter": 0, "interaction": 0,"constraints_num":0, "ari": 0}]]
    skeleton, order = order_allocation(skeleton, representative)
    # 共享数据结构
    manager = multiprocessing.Manager()
    query_times = manager.Value('i', 0)
    constraints_num = manager.Value('i', 0)
    # 进程锁
    user_locks = manager.dict({user.user_id: manager.Lock() for user in users_list})
    query_times_lock = manager.Lock()
    constraints_num_lock = manager.Lock()
    neighbourhoods_lock = manager.Lock()
    label_dict_lock = manager.Lock()
    rest_point_lock = manager.Lock()
    # ------------------------------------
    try:
        with multiprocessing.Pool(processes=max(2, multiprocessing.cpu_count() // 2)) as pool:
            neighborhood,neighborhood_r,neighborhood_r_behind,count,order,users_list,constraints_num=neighborhood_initialization(data, order, representative, real_labels,users_list,user_locks,min_users_num,
                                                                                                                                 max_users_num,max_uncertainty_num,pool,query_times_lock,constraints_num_lock,neighbourhoods_lock,
                                                                                                                                 query_times,constraints_num,isUpdate)

            predict_labels = influence_model_propagation(skeleton, neighborhood)
            ari=adjusted_rand_score(real_labels, predict_labels)
            record.append([{"iter": 1, "interaction": count,"constraints_num":constraints_num, "ari": ari}])
            record = neighborhood_learning(skeleton, data, predict_labels, neighborhood, k_nearest_neighbors, count, order,
                                           real_labels, record,k,users_list,user_locks,min_users_num, max_users_num,max_uncertainty_num,constraints_num
                                           ,pool,query_times_lock,constraints_num_lock,neighbourhoods_lock,label_dict_lock,rest_point_lock,manager,isUpdate)
    except KeyboardInterrupt:
        print("进程被中断，正在终止子进程...")
        pool.terminate()  # 强制终止所有子进程
        pool.join()  # 等待子进程彻底退出
        print("子进程已终止，安全退出。")

    except Exception as e:
        print("发生异常:", e)
        traceback.print_exc()  # 打印异常信息，方便调试
        pool.terminate()  # 终止子进程，防止泄漏
        pool.join()
    finally:
        if pool:
            pool.close()
            pool.join()
        manager.shutdown()  # 释放 manager 及其管理的所有资源
    return record




