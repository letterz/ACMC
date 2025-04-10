import numpy as np


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
    # feature_nums=data_matrix.shape[1]
    # # 使用列表推导式生成列名
    # column_names = [f'feature_{i}' for i in range(feature_nums)]
    # #得到dataframe
    # df = pd.DataFrame(data=data_matrix, columns=column_names)

    return data_matrix,true_label