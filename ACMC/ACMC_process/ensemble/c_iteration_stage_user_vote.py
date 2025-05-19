from .user import distribute_node_pair_to_users
from .a_pre_cluster import query_process, update_user_confidence_process,\
    pre_cluster_user_vote_process,get_gamma_ij



def process_xi(xi, query_list_dict, users_list, true_label, user_locks, min_users_num, max_users_num,
               query_times_shared, constraints_num_shared, neighbourhoods_list_shared, max_uncertain_xi_list_label_dict,
               rest_point, query_times_lock, constraints_num_lock, neighbourhoods_lock, label_dict_lock,
               rest_point_lock):
    try:
        flag = 0
        xi_query_list = query_list_dict[xi]

        for one_tuple in xi_query_list:
            min_max = (min(xi, one_tuple[0]), max(xi, one_tuple[0]))
            seed = hash(min_max)

            distributed_user_list = distribute_node_pair_to_users(users_list, min_users_num, max_users_num, seed=seed)
            voting_result = query_process(min_max[0], min_max[1], distributed_user_list, true_label, user_locks)

            
            with query_times_lock:
                query_times_shared.value += len(distributed_user_list)
            with constraints_num_lock:
                constraints_num_shared.value += 1
            vote_0_userlist, vote_1_userlist, gamma_ij = get_gamma_ij(distributed_user_list, voting_result, user_locks)
            if gamma_ij == 0:
                update_user_confidence_process(distributed_user_list, punish_user_index=vote_1_userlist,
                                              user_locks=user_locks, isUpdate=True)
                flag = 0
            else:
                if one_tuple[1] >= len(neighbourhoods_list_shared):
                    print(
                        f"Index {one_tuple[1]} out of range for neighbourhoods_list_shared with length {len(neighbourhoods_list_shared)}")
                else:
                    
                    with neighbourhoods_lock:
                        temp_list = neighbourhoods_list_shared[:]  
                        temp_list[one_tuple[1]].append(xi)  
                        neighbourhoods_list_shared[:] = temp_list  
                        
                    with label_dict_lock:
                        temp_dict = dict(max_uncertain_xi_list_label_dict)  
                        temp_dict[xi] = one_tuple[1] 
                        max_uncertain_xi_list_label_dict.clear() 
                        max_uncertain_xi_list_label_dict.update(temp_dict) 

                    update_user_confidence_process(distributed_user_list, punish_user_index=vote_0_userlist,
                                                  user_locks=user_locks, isUpdate=True)
                    flag = 1
                    break

        if flag == 0:
            with rest_point_lock:
                rest_point.append(xi)
               
       
    except Exception as e:
        print(f"process_xi error for xi={xi}: {e}")
    return 0

def iteration_stage_user_vote_process(max_uncertain_xi_list, query_list_dict, users_list, true_label, query_times,
                                     neighbourhoods_list, min_users_num, max_users_num, user_locks, constraints_num
                                     ,pool,query_times_lock,constraints_num_lock,neighbourhoods_lock,label_dict_lock,rest_point_lock,manager,isUpdate):
    try:
        max_uncertain_xi_list_label_dict = manager.dict()
        rest_point = manager.list()
        query_times_shared = manager.Value('i', query_times)
        constraints_num_shared = manager.Value('i', constraints_num)
        neighbourhoods_list_shared = manager.list(neighbourhoods_list)
        try:
            futures = [pool.apply_async(process_xi, args=(xi, query_list_dict, users_list, true_label, user_locks,
                                                       min_users_num, max_users_num, query_times_shared,
                                                       constraints_num_shared, neighbourhoods_list_shared,
                                                       max_uncertain_xi_list_label_dict, rest_point,
                                                       query_times_lock,
                                                       constraints_num_lock, neighbourhoods_lock, label_dict_lock,
                                                       rest_point_lock))
                for xi in max_uncertain_xi_list]
            for future in futures:
                future.get()

        except Exception as e:
            print(f"error: {e}")
            raise 

        # Process the remaining points.
        temp_nei, users_list, query_times, constraints_num = pre_cluster_user_vote_process(
            list(rest_point), users_list, true_label, min_users_num, max_users_num, query_times_shared, user_locks,
            constraints_num_shared,pool,query_times_lock,constraints_num_lock,neighbourhoods_lock, isUpdate)
        with neighbourhoods_lock:
            for one_nei in temp_nei:
                neighbourhoods_list_shared.append(one_nei)
    except KeyboardInterrupt:
        print("Debugging interrupted; releasing resources...")
        raise  

    return list(neighbourhoods_list_shared), users_list, query_times_shared.value, dict(
        max_uncertain_xi_list_label_dict), constraints_num_shared.value



