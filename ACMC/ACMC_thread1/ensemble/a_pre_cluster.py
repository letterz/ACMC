import math

import numpy as np
import concurrent
import threading

import random
from .user import user_judge_func,distribute_node_pair_to_users


def pre_cluster_user_vote_thread(points_list, users_list, true_label, min_users_num, max_users_num, query_times,user_locks,constraints_num,isUpdate):
    Nei = []  # 邻域
    lock = threading.Lock()  # 用于保护共享数据的锁
    query_times_lock = threading.Lock()  # 专门为 query_times 添加的锁
    constraints_num_lock=threading.Lock()
    unfind_points_lock = threading.Lock()  # 用于保护 unfind_points_list 的锁

    def process_point(i, first_point_of_nei):
        nonlocal query_times  # 允许修改外部的 query_times
        nonlocal constraints_num
        min_max = (min(first_point_of_nei, points_list[i]), max(first_point_of_nei, points_list[i]))
        seed = hash(min_max)  # 使用点对生成唯一的 seed
        # 把点对随机分配给几个用户
        distributed_user_list = distribute_node_pair_to_users(users_list, min_users_num, max_users_num, seed=seed)
        voting_result = query_thread(min_max[0], min_max[1], distributed_user_list, true_label,user_locks)
        # 使用锁来保护 query_times 的更新
        with query_times_lock:
            query_times += len(distributed_user_list)
        with constraints_num_lock:
            constraints_num+=1

        vote_0_userlist, vote_1_userlist,gamma_ij=get_gamma_ij(distributed_user_list, voting_result, user_locks)
        if gamma_ij == 0:
            # 说明两点为 cannot-link
            update_user_confidence_thread(distributed_user_list, punish_user_index=vote_1_userlist,
                                          user_locks=user_locks, isUpdate=isUpdate)
            # 使用锁保护 unfind_points_list 的修改
            with unfind_points_lock:
                unfind_points_list.append(points_list[i])
            return False  # 该点与当前邻域不相连
        else:
            # 说明两点为 must-link
            update_user_confidence_thread(distributed_user_list, punish_user_index=vote_0_userlist,
                                          user_locks=user_locks, isUpdate=isUpdate)
            return True  # 该点与当前邻域相连

    while len(points_list) != 0:
        Nei.append([points_list[0]])  # 创建新邻域，并将当前点作为第一个点
        first_point_of_nei = points_list[0]  # 当前邻域的第一个点
        unfind_points_list = []  # 用于存储无法加入当前邻域的点

        # 多线程并行处理 points_list 中的点
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_point = {executor.submit(process_point, i, first_point_of_nei): i for i in range(1, len(points_list))}
            for future in concurrent.futures.as_completed(future_to_point):
                i = future_to_point[future]
                try:
                    result = future.result()
                    if result:
                        # 如果该点可以加入当前邻域，使用锁进行保护
                        with lock:
                            Nei[-1].append(points_list[i])
                except Exception as exc:
                    print(f'Error occurred when processing point {i}: {exc}')

        # 更新 points_list 为未找到的点，继续处理
        points_list = unfind_points_list[:]

    return Nei, users_list, query_times,constraints_num

def query_thread(point_a, point_b, user_list, true_label, user_locks):
    user_num = len(user_list)
    voting_results = [-1] * user_num
    # 每个用户投票
    for i in range(0, user_num):
        user = user_list[i]
        with user_locks[user]:
            user_result = user_judge_func(point_a, point_b, user, true_label)
            user.query_times+=1#用户查询次数加一
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

    # ------------------------
    # 如果约束没有发生冲突,且所有人都投must
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
    else:#大家意见不一致
        sum_of_gamma_ij_u_multiple_Cu,sum_one_subtrac_gamma_ij_u_multiple_Cu=cul_gammaij_mul_Cu_sum_and_one_minus_sum(voting_result,users_list)
        delta_ij=(one_counts*sum_of_gamma_ij_u_multiple_Cu)/(zero_counts*sum_one_subtrac_gamma_ij_u_multiple_Cu)
        gamma_ij=np.floor(2 / (1 + np.exp(1 - delta_ij)))

    return  vote_0_userlist, vote_1_userlist,gamma_ij
def update_user_confidence_thread(users_list, punish_user_index, user_locks,isUpdate=True):
    """
    :param users_list:
    :param punish_user_index: 被惩罚的用户下标列表
    :param reward_user_index: 被奖励的用户下标列表
    :param user_locks: 用户锁字典，保证多线程更新置信度时不会冲突
    """
    #更新置信度
    if isUpdate==True:
        # 惩罚投错的用户，修改他们的置信度
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
        pass#不更新置信度
    return 0