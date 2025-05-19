import math

import numpy as np
import concurrent
import threading

import random
from .user import user_judge_func,distribute_node_pair_to_users


def pre_cluster_user_vote_thread(points_list, users_list, true_label, min_users_num, max_users_num, query_times,user_locks,constraints_num,isUpdate):
    Nei = []  
    lock = threading.Lock()  
    query_times_lock = threading.Lock()  
    constraints_num_lock=threading.Lock()
    unfind_points_lock = threading.Lock() 

    def process_point(i, first_point_of_nei):
        nonlocal query_times  
        nonlocal constraints_num
        min_max = (min(first_point_of_nei, points_list[i]), max(first_point_of_nei, points_list[i]))
        seed = hash(min_max) 
        
        distributed_user_list = distribute_node_pair_to_users(users_list, min_users_num, max_users_num, seed=seed)
        voting_result = query_thread(min_max[0], min_max[1], distributed_user_list, true_label,user_locks)
        
        with query_times_lock:
            query_times += len(distributed_user_list)
        with constraints_num_lock:
            constraints_num+=1

        vote_0_userlist, vote_1_userlist,gamma_ij=get_gamma_ij(distributed_user_list, voting_result, user_locks)
        if gamma_ij == 0:
           
            update_user_confidence_thread(distributed_user_list, punish_user_index=vote_1_userlist,
                                          user_locks=user_locks, isUpdate=isUpdate)
           
            with unfind_points_lock:
                unfind_points_list.append(points_list[i])
            return False 
        else:
           
            update_user_confidence_thread(distributed_user_list, punish_user_index=vote_0_userlist,
                                          user_locks=user_locks, isUpdate=isUpdate)
            return True 

    while len(points_list) != 0:
        Nei.append([points_list[0]]) 
        first_point_of_nei = points_list[0] 
        unfind_points_list = [] 

       
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_point = {executor.submit(process_point, i, first_point_of_nei): i for i in range(1, len(points_list))}
            for future in concurrent.futures.as_completed(future_to_point):
                i = future_to_point[future]
                try:
                    result = future.result()
                    if result:
                       
                        with lock:
                            Nei[-1].append(points_list[i])
                except Exception as exc:
                    print(f'Error occurred when processing point {i}: {exc}')

        points_list = unfind_points_list[:]

    return Nei, users_list, query_times,constraints_num

def query_thread(point_a, point_b, user_list, true_label, user_locks):
    user_num = len(user_list)
    voting_results = [-1] * user_num
   
    for i in range(0, user_num):
        user = user_list[i]
        with user_locks[user]:
            user_result = user_judge_func(point_a, point_b, user, true_label)
            user.query_times+=1
            voting_results[i] = user_result

    voting_results = [int(item) for item in voting_results]
    return voting_results

def get_gamma_ij(users_list, voting_result, user_locks):
    from collections import Counter
    counts = Counter(voting_result)
    zero_counts = counts[0]
    one_counts = counts[1]
    vote_0_userlist = [i for i, v in enumerate(voting_result) if v == 0]
    vote_1_userlist = [i for i, v in enumerate(voting_result) if v == 1]

    def cul_gammaij_mul_Cu_sum_and_one_minus_sum(voting_result,users_list):
        sum_gamma_ij_u_multiple_Cu = 0
        sum_one_subtrac_gamma_ij_u_multiple_Cu = 0
        for i in range(len(voting_result)):
            sum_gamma_ij_u_multiple_Cu+=voting_result[i]*users_list[i].confidence
            sum_one_subtrac_gamma_ij_u_multiple_Cu+=(1-voting_result[i])*users_list[i].confidence
        return sum_gamma_ij_u_multiple_Cu,sum_one_subtrac_gamma_ij_u_multiple_Cu
    gamma_ij=None
    if zero_counts==0 and one_counts==len(voting_result):
        gamma_ij=1
    elif one_counts==0 and zero_counts==len(voting_result):
        gamma_ij=0
    else:
        sum_of_gamma_ij_u_multiple_Cu,sum_one_subtrac_gamma_ij_u_multiple_Cu=cul_gammaij_mul_Cu_sum_and_one_minus_sum(voting_result,users_list)
        delta_ij=(one_counts*sum_of_gamma_ij_u_multiple_Cu)/(zero_counts*sum_one_subtrac_gamma_ij_u_multiple_Cu)
        gamma_ij=np.floor(2 / (1 + np.exp(1 - delta_ij)))

    return  vote_0_userlist, vote_1_userlist,gamma_ij
def update_user_confidence_thread(users_list, punish_user_index, user_locks,isUpdate=True):

    if isUpdate==True:
       
        for index in punish_user_index:
            with user_locks[users_list[index]]:
                users_list[index].error_times += 1
                if users_list[index].query_times<=15:
                    pass
                else:
                    if users_list[index].isExpert==True:
                        pass
                    else:
                        users_list[index].confidence = 1 - (users_list[index].error_times / users_list[index].query_times)
    else:
        pass
    return 0
