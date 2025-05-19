
from heapq import nsmallest
from operator import itemgetter
from scipy.spatial.distance import euclidean

def construct_query_list_for_max_uncertain_xi_list(max_uncertain_xi_list,  N,data):

    query_list_dict={}
    for xi in max_uncertain_xi_list:
       
        xi_query_list=construct_query_list_Q_for_xi(xi, N,data)
        query_list_dict[xi]=xi_query_list
    
    return query_list_dict

def construct_query_list_Q_for_xi(max_uncertain_xi, neighborhood_N, data):

    closest_x_list = []  
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

 
    closest_x_list.sort(key=lambda x: x[2])

    closest_x = [(item[0], item[1]) for item in closest_x_list]
    return closest_x
