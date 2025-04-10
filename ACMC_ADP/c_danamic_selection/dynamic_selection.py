

import numpy as np


def uncertainty_selection_optimized(predict_labels, data, neighbors, result_dict, nearest_neighbors, max_un_ins_num):
    # 将 predict_labels 转换为 NumPy 数组，以支持多索引操作
    predict_labels = np.array(predict_labels)

    # 扁平化邻居列表
    flattened_neighbors_set = set([item for sublist in neighbors for item in sublist])  # 使用 set 提高查找效率

    # 初始化不确定度向量
    uncertainty_vec = []

    for i in range(len(data)):
        if i not in flattened_neighbors_set:
            # 计算分母：邻居数量
            neighbor_indices = nearest_neighbors[i]
            denominator = len(neighbor_indices)

            if denominator == 0:
                continue  # 避免除以零

            # 计算每个类别的分子
            neighbor_labels = predict_labels[neighbor_indices]  # 现在可以使用多个索引
            label_counts = np.bincount(neighbor_labels, minlength=len(result_dict))

            # 计算熵
            proportions = label_counts / denominator
            non_zero_proportions = proportions[proportions > 0]  # 去除 0 以避免 log2 出现问题
            entropy = -np.sum(non_zero_proportions * np.log2(non_zero_proportions))

            # 将结果保存到不确定度向量中
            uncertainty_vec.append([i, entropy])

    if uncertainty_vec:
        uncertainty_vec = np.array(uncertainty_vec)
        sorted_indices = np.argsort(uncertainty_vec[:, 1])[::-1]  # 按不确定度从高到低排序
        top_m_points = uncertainty_vec[sorted_indices[:max_un_ins_num], 0].astype(int)  # 取前 m 个点的索引
    else:
        top_m_points = []
    # print(f'uncertainty_selection_optimized 执行完毕')

    return top_m_points








