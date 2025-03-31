import csv
import math
import os
import time
import numpy as np
from ACMC.ACMC_process.ensemble.user import get_error_rate_list,create_some_users
from ACMC.ACMC_process.e_neighborhood_learning import ACMC_process
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

def result_to_csv_ACMC(ARI_record, title,output_path):
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
def run_ACMC_process(output_path,algo_name,large_or_small,repetitions_times,error_span,min_users_n, max_users_n):
    print('----------开始执行ACDM_process------')
    datasets = [('iris', 'iris.csv',)]
    output_path = os.path.join(output_path,algo_name,large_or_small,)
    for index in range(0,1):
        datasets_path = os.path.join("datasets", large_or_small, datasets[index][0], datasets[index][1])
        title=datasets[index][0]
        print(title)
        data, real_labels = get_data_from_datasets(datasets_path)
        data = data_preprocess(data)
        k = 10
        #用户数目和不确定点数目
        beta=cul_uncertainty_num(data.shape[0], 2, 1,)
        usernum = beta
        print(f'k={k},max_uncertainty_num={beta},usernum={usernum},min_users_n={min_users_n}, max_users_n={max_users_n}')
        error_rate_list = get_error_rate_list(usernum, error_span)
        print(f'error={error_rate_list}')
        min_users_num, max_users_num = min_users_n, max_users_n
        temp_output_path=os.path.join(output_path,f'error={error_rate_list[0]}_{error_rate_list[-1]}_{error_rate_list[-2]}_users={min_users_num}_{max_users_num}')

        users_list = create_some_users(usernum, error_rate_list, 1)
        for re_times in range(0,repetitions_times):
            print(f'第{re_times + 1}次运行{title}')
            final_path = os.path.join(temp_output_path, f'{re_times + 1}')
            start_time = time.time()
            ARI_record=ACMC_process(data, real_labels,k,users_list,min_users_num, max_users_num,beta)
            end_time = time.time()
            print(f'{end_time - start_time}s')
            print(f'final ari={ARI_record[-1]}')
            record_run_time(end_time - start_time, datasets[index][0], temp_output_path)
            result_to_csv_ACMC(ARI_record, datasets[index][0], final_path)
            print(f'写入路径={final_path}')
            print(f'{datasets[index][0]} finish')

    print('ACDM_process 结束')
    return 0

if __name__ == '__main__':

    output_path= 'output/test_2'
    large_or_small, repetitions_times='small',1
    # run_ACMC_process(output_path,'ACMC_process',large_or_small,repetitions_times
    #                 ,error_span=0.05,min_users_n=2, max_users_n=3)
    run_ACMC_process(output_path, 'ACMC_process', large_or_small, repetitions_times
                     , error_span=0, min_users_n=1, max_users_n=1)