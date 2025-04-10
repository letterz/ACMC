
import csv
import math
import os
import time
import threading

import numpy as np
import pandas as pd
from ACMC_ADP.d_experiment.multi_noisy_experience_adp import (experiemnt_ACMC_adp_thread, )
from ACMC_ADP.ensemble.user import get_error_rate_list,create_some_users


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
def data_preprocess(data):
    size=np.shape(data)
    random_matrix=np.random.rand(size[0],size[1]) * 0.000001
    data=data+random_matrix
    return data
def record_run_time(time, dataset_name, output_path):
    wirte_path=os.path.join(output_path,'time')
    os.makedirs(wirte_path, exist_ok=True)
    output_file=os.path.join(wirte_path, f'{dataset_name}_runtime.csv')
    with open(output_file,'a') as file:
        file.write(f"{time}\n")
    return 0


def result_to_csv_ADP_thread(ARI_record, title,output_path):
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
def cul_uncertainty_num(value,base,alpha):
    num=math.ceil(alpha*math.log(value, base))
    return num

def set_different_error_rate1(usernum,p_type,ratio_type,common_errorrate):
    errorrate=[]
    if p_type=="expert":
        zhuanjia_num=1+math.floor(math.log(usernum, 10))
        print(f"专家人数={zhuanjia_num}")
        zhuanjia_error_list=[]
        for i in range(zhuanjia_num):
            zhuanjia_error_list.append(0)
        other_num=usernum-zhuanjia_num
        other_error_list=[0.05 for i in range(other_num)]
        errorrate=zhuanjia_error_list+other_error_list
    if p_type=="elite":
        jingying_num=1+math.floor(math.log(usernum, 10))
        print(f"精英人数={jingying_num}")
        #精英错误率为0.02
        for i in range(usernum):
            if i<jingying_num:
                errorrate.append(0.02)
            else:
                errorrate.append(0.05)
    if p_type=="common":
        errorrate=[0.05 for i in range(usernum)]
    return errorrate
def run_ACMC_ADP_thread(output_path,algo_name,large_or_small,repetitions_times,error_span,min_users_n, max_users_n,isUpdate=None, ):
    print('------开始执行ACMC_ADP_thread---------')
    # 1. 从数据集中读取数据
    datasets_names1 = [('iris', 'iris.csv',), ('fertility', 'fertility.csv',), ('sonar', 'sonar.csv',),
                       ('seeds', 'seeds.csv',), ('haberman', 'haberman.csv',), ('ionosphere', 'ionosphere.csv',),
                       ('musk', 'musk.csv',), ('balance', 'balance.csv',), ('breast', 'breast.csv',),
                       ('pima', 'pima.csv',), ('vehicle', 'vehicle.csv',), ('mfeat_karhunen', 'mfeat_karhunen.csv',), ]
    # datasets_names1 =[('EEG', 'EEG.csv'), ]
    output_path=os.path.join(output_path,algo_name,large_or_small)

    for index in range(0, 1):
        datasets_path = os.path.join("../../datasets", large_or_small, datasets_names1[index][0], datasets_names1[index][1])
        print(f"-----------{datasets_names1[index][0]}数据集------------")
        alpha = 0.22
        l = 5
        theta = 0.00001
        data, real_labels = get_data_from_datasets(file_path=datasets_path)
        max_uncertain_xi_num = cul_uncertainty_num(data.shape[0], 2, 1, )
        # 创建用户
        usernum = max_uncertain_xi_num
        print(f'usernum=max_uncertain_xi_num={max_uncertain_xi_num}')
        error_rate_list = get_error_rate_list(usernum, span=error_span)
        min_users_num, max_users_num = min_users_n, max_users_n
        users_list = create_some_users(usernum, error_rate_list, 1)
        # 初始化用户锁
        user_locks = {user: threading.Lock() for user in users_list}
        temp_output_path = os.path.join(output_path, f'error={error_rate_list[0]}_{error_rate_list[1]}_{error_rate_list[2]}_users={min_users_num}_{max_users_num}')
        print(f'error={error_rate_list}')
        for re_times in range( repetitions_times):
            print(f'第{re_times + 1}次运行{datasets_names1[index][0]}')
            final_path = os.path.join(temp_output_path, f'{re_times + 1}')

            start_time = time.time()
            ARI_record = experiemnt_ACMC_adp_thread(data, real_labels, alpha, l, theta, users_list, min_users_num, max_users_num,
                                        max_uncertain_xi_num, user_locks,isUpdate)
            end_time = time.time()
            print(f'{ARI_record[-1]}')
            record_run_time(end_time - start_time, datasets_names1[index][0], temp_output_path)
            # result_to_csv(ARI_record, datasets_names1[index][0],output_path)
            result_to_csv_ADP_thread(ARI_record, datasets_names1[index][0], final_path)
            print(f'{datasets_names1[index][0]} finish')

    return 0


if __name__ == '__main__':

    output_path='output'
    large_or_small, repetitions_times='small',1
    run_ACMC_ADP_thread(output_path, 'ACMC_ADP_thread', large_or_small, repetitions_times,
                    error_span=0, min_users_n=1, max_users_n=1, isUpdate=False, )