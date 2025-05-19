
import numpy as np

from .user import user_judge_func,distribute_node_pair_to_users

def process_point(i, first_point_of_nei, points_list, users_list, true_label, min_users_num, max_users_num, query_times, user_locks, constraints_num,query_times_lock,constraints_num_lock, isUpdate):
    min_max = (min(first_point_of_nei, points_list[i]), max(first_point_of_nei, points_list[i]))
    seed = hash(min_max)

    distributed_user_list = distribute_node_pair_to_users(users_list, min_users_num, max_users_num, seed=seed)
    voting_result = query_process(min_max[0], min_max[1], distributed_user_list, true_label, user_locks)

    with query_times_lock:
        query_times.value += len(distributed_user_list)
    with constraints_num_lock:
        constraints_num.value += 1

    vote_0_userlist, vote_1_userlist,gamma_ij=get_gamma_ij(distributed_user_list, voting_result, user_locks)
    if gamma_ij==0:
        update_user_confidence_process(distributed_user_list, vote_1_userlist, user_locks, isUpdate)
        return (i, False)
    else:
        update_user_confidence_process(distributed_user_list, vote_0_userlist, user_locks, isUpdate)
        return (i, True)

def pre_cluster_user_vote_process(points_list, users_list, true_label, min_users_num, max_users_num, query_times, user_locks, constraints_num, pool,query_times_lock,constraints_num_lock,neighbourhoods_lock,isUpdate):

    Nei = []
    try:
        while points_list:
            Nei.append([points_list[0]])
            first_point_of_nei = points_list[0]
            unfind_points_list = []

            results = []
            for i in range(1, len(points_list)):
                async_result = pool.apply_async(
                    process_point,
                    (i, first_point_of_nei, points_list, users_list, true_label, min_users_num, max_users_num,
                     query_times, user_locks, constraints_num, query_times_lock, constraints_num_lock, isUpdate)
                )
                results.append((i, async_result))

            
            for i, async_result in results:
                try:
                    i, result = async_result.get()
                    if result:
                        with neighbourhoods_lock:
                            Nei[-1].append(points_list[i])
                    else:
                        unfind_points_list.append(points_list[i])
                except KeyboardInterrupt:
                    print("KeyboardInterrupt detected, terminating process pool...")
                    pool.terminate()  
                    pool.join()  
                    raise  

            points_list = unfind_points_list[:]
    finally:
        pass  
    return Nei, users_list, query_times.value, constraints_num.value

def query_process(point_a, point_b, user_list, true_label, user_locks):
    voting_results = []
    for user in user_list:
        with user_locks[user.user_id]:
            user_result = user_judge_func(point_a, point_b, user, true_label)
            user.query_times += 1
            voting_results.append(user_result)
    return [int(v) for v in voting_results]


def get_gamma_ij(users_list, voting_result, user_locks):
    from collections import Counter
    counts = Counter(voting_result)
    zero_counts = counts[0]
    one_counts = counts[1]
    vote_0_userlist = [i for i, v in enumerate(voting_result) if v == 0]
    vote_1_userlist = [i for i, v in enumerate(voting_result) if v == 1]

    # ------------------------
    # If there are no conflicts in the constraints and everyone votes "must",
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
    else:#People have differing opinions.
        sum_of_gamma_ij_u_multiple_Cu,sum_one_subtrac_gamma_ij_u_multiple_Cu=cul_gammaij_mul_Cu_sum_and_one_minus_sum(voting_result,users_list)
        delta_ij=(one_counts*sum_of_gamma_ij_u_multiple_Cu)/(zero_counts*sum_one_subtrac_gamma_ij_u_multiple_Cu)
        gamma_ij=np.floor(2 / (1 + np.exp(1 - delta_ij)))

    return  vote_0_userlist, vote_1_userlist,gamma_ij

def update_user_confidence_process(users_list, punish_user_index, user_locks, isUpdate=True):
    if isUpdate:
        for index in punish_user_index:
            with user_locks[users_list[index].user_id]:
                users_list[index].error_times += 1
                if users_list[index].query_times > 15 and not users_list[index].isExpert:
                    users_list[index].confidence = 1 - (users_list[index].error_times / users_list[index].query_times)
