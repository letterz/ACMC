import threading
from concurrent.futures import ThreadPoolExecutor
from .user import distribute_node_pair_to_users
from .a_pre_cluster import query_thread,get_gamma_ij,update_user_confidence_thread,pre_cluster_user_vote_thread
def iteration_stage_user_vote_thread(max_uncertain_xi_list, query_list_dict, users_list, true_label, query_times,
                                                 neighbourhoods_list, min_users_num, max_users_num,user_locks,constraints_num,isUpdate ):
    """对用户也加上锁了"""
    # Mapping of points to their labels after voting
    max_uncertain_xi_list_label_dict = {}
    rest_point = []

    # Create locks for thread safety
    query_times_lock = threading.Lock()
    constraints_num_lock = threading.Lock()
    neighbourhoods_lock = threading.Lock()
    label_dict_lock = threading.Lock()
    rest_point_lock = threading.Lock()

    # # 初始化用户锁
    # user_locks = {user: threading.Lock() for user in users_list}
    def process_xi(xi):
        nonlocal query_times
        nonlocal constraints_num
        flag = 0
        xi_query_list = query_list_dict[xi]

        for one_tuple in xi_query_list:
            min_max = (min(xi, one_tuple[0]), max(xi, one_tuple[0]))
            seed = hash(min_max)

            # Distribute the point pair to users
            distributed_user_list = distribute_node_pair_to_users(users_list, min_users_num, max_users_num, seed=seed)
            voting_result = query_thread(min_max[0], min_max[1], distributed_user_list, true_label, user_locks)

            # Update query times safely
            with query_times_lock:
                query_times += len(distributed_user_list)
            with constraints_num_lock:
                constraints_num += 1


            vote_0_userlist, vote_1_userlist, gamma_ij = get_gamma_ij(distributed_user_list, voting_result, user_locks)
            if gamma_ij == 0:
                update_user_confidence_thread(distributed_user_list, punish_user_index=vote_1_userlist,
                                              user_locks=user_locks, isUpdate=isUpdate)
                flag = 0
            else:
                with neighbourhoods_lock:
                    neighbourhoods_list[one_tuple[1]].append(xi)

                with label_dict_lock:
                    max_uncertain_xi_list_label_dict[xi] = one_tuple[1]

                update_user_confidence_thread(distributed_user_list, punish_user_index=vote_0_userlist,
                                              user_locks=user_locks, isUpdate=isUpdate)
                flag = 1
                break

        if flag == 0:
            # Append to rest_point safely
            with rest_point_lock:
                rest_point.append(xi)

    # Use ThreadPoolExecutor to parallelize processing of points
    with ThreadPoolExecutor() as executor:
        executor.map(process_xi, max_uncertain_xi_list)

    # Process rest points after voting
    temp_nei, users_list, query_times,constraints_num = pre_cluster_user_vote_thread(rest_point, users_list, true_label, min_users_num, max_users_num, query_times,user_locks,constraints_num,isUpdate)

    # Add temp_nei neighborhoods to the main list safely
    with neighbourhoods_lock:
        for one_nei in temp_nei:
            neighbourhoods_list.append(one_nei)

    return neighbourhoods_list, users_list, query_times, max_uncertain_xi_list_label_dict,constraints_num


