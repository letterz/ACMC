import random


class user:
    def __init__(self,user_id,username,confidence,judged_constraint=None,
                 error_rate=None,judge_times=None,will_judge_constraint=None,
                 query_times=None,error_times=None,current_acc_rate=None,):#random.random()
        """
        :param username:
        :param confidence: 用户置信度
        :param error_constraint: 从数据集中生成的，用户的错误约束
        :param constraint: 用户判断过的所有约束
        :param error_rate:用户错误率
        """
        self.user_id=user_id
        self.username=username
        self.confidence = confidence
        self.error_rate= error_rate

        self.query_times=query_times
        self.error_times=error_times
        self.current_acc_rate=current_acc_rate

        if judged_constraint is None:
            judged_constraint={}
        self.judged_constraint=judged_constraint

        if judge_times is None:
            judge_times=0
        self.judge_times = judge_times

        if will_judge_constraint is None:
            will_judge_constraint={}
        self.will_judge_constraint=will_judge_constraint

        if query_times is None:
            query_times=0

        if error_times is None:
            correct_times=0

        if self.error_rate==0:
            self.isExpert=True
        else:
            self.isExpert = False
    def print_message(self):
        print(f"用户 {self.username} 的置信度confidence={self.confidence}，错误率error_rate={self.error_rate}")

def create_some_users(user_num,user_error_rate,initial_confidence,):
    users_list=[]
    for i in range(0,user_num):
        # newuser=user(name=i,error_rate=random.uniform(0, 0.05))
        newuser_name="user"+str(i+1)
        newuser = user(user_id=i,username=newuser_name, confidence=initial_confidence,error_rate=user_error_rate[i],
        query_times=0,error_times=0,current_acc_rate=0)

        if newuser.error_rate==0:
            newuser.current_acc_rate=1
        users_list.append(newuser)
    return users_list


def get_error_rate_list(num,span):
    return [span for i in range(num)]
def distribute_node_pair_to_users(users_list,min_users_num,max_users_num,seed):
    # random.seed(seed)
    num=random.randint(min_users_num, max_users_num)
    distributed_user_list=random.sample(users_list, num)
    return distributed_user_list

def is_user_error_constraint(user, a, b, true_label):
    """
    按需查询用户对点对 (a, b) 是否存在错误约束。
    :param user: 用户对象，包含 error_rate 属性
    :param a: 点 a 的索引
    :param b: 点 b 的索引
    :param n: 数据集中点的总数
    :param true_label: 数据真实标签
    :return: 0 表示 cannot-link (错误)，1 表示 must-link (错误)，None 表示正确约束
    """
    # 保证用户查询同样的点对时结果一致
    # random.seed(hash((user.user_id, a, b)))

    # 计算总点对数

    # 根据用户的错误率计算是否为错误约束
    if random.random() < user.error_rate:
        # 该点对是错误约束
        if true_label[a] == true_label[b]:
            return 0  # 本来是 must-link，返回错误的 cannot-link
        else:
            return 1  # 本来是 cannot-link，返回错误的 must-link
    else:
        # 该点对不是错误约束，返回 None 表示正确约束
        return None
def user_judge_func(point_a,point_b,user,true_label):
    #查看约束是否在error_constraints字典里面，如果在里面则把字典里面的值作为约束结果，如果不在里面,说明在这个约束上用户没有犯错,则从真实标签里得到结果
    judges_result=is_user_error_constraint(user,point_a,point_b,true_label)
    if judges_result is not None:
        user_result=judges_result

    else:
        if true_label[point_a]==true_label[point_b]:
            user_result=1
        else:
            user_result=0

    return user_result