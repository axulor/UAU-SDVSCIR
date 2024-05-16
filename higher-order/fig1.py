import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
import copy
import scipy.sparse as sp
import numpy as np
from Epidemic import Epidemic


# 依据物理层网络生成意识层网络
def add_random_edges(G, num):
    """
    向原始网络中添加指定数目的随机连边生成新网络
    :param G: 原始网络
    :param num: 指定随机连边数目
    :return: 生成网络H
    """
    H = copy.deepcopy(G)
    non_edges = list(nx.non_edges(H))
    edges = np.random.choice(len(non_edges) - 1, size=num, replace=False)
    add_edges = []
    for i in edges:
        add_edges.append(non_edges[i])
    H.add_edges_from(add_edges)

    return H

def netsimplicial_random(N, ntri):
    '''
    生成一个随机简单网络，通过随机连接节点来形成新的三角形。

    :param N: int
        网络中的总节点数。
    :param ntri: int
        每次添加新节点时，通过随机选择现有的边来与新节点连接，形成的三角形数量。
    :return: scipy.sparse.dok_matrix
        生成的网络的邻接矩阵，使用稀疏矩阵格式。
    '''

    mlinks = 2 * ntri
    N0 = mlinks + 1
    A = sp.dok_matrix((N, N), dtype=np.float32)
    A[0:N0, 0:N0] = 1
    A.setdiag(0)

    for n in range(N0, N):
        i, j = sp.triu(A, 1).nonzero()
        l = len(i)
        isw = 0
        while isw == 0:
            m = np.random.choice(l, ntri, replace=False)
            v = np.concatenate((i[m], j[m]))
            if len(np.unique(v)) == 2 * ntri:
                isw = 1
        for k in m:
            A[n, i[k]] = 1
            A[i[k], n] = 1
            A[n, j[k]] = 1
            A[j[k], n] = 1
    G = nx.from_scipy_sparse_array(A)
    return G

# 为了执行模拟，首先我们需要定义一个函数来执行单次模拟并收集数据
def run_simulation(tau_value, lower_net, upper_net):
    # 创建新的流行病模型实例，使用指定的tau值
    epidemic_model_tau = Epidemic(
        lower_net=lower_net,
        upper_net=upper_net,
        alpha=0.5,
        alpha_triangle=1.0,
        delta=0.5,
        delta_triangle=1.0,
        tau=tau_value,  # 使用函数参数中的tau值
        beta=0.4,
        eff=0.8,
        omega=0.1,
        omega_triangle=1.0,
        eta=0.6,
        gamma=0.3333
    )

    # 初始化网络状态
    epidemic_model_tau.init_awareness(init_u=0.5)
    epidemic_model_tau.init_strategy(init_c=0.1)
    epidemic_model_tau.init_state()
    epidemic_model_tau.init_infect(init_i=0.1)

    # 存储每个状态的密度
    results = {"US_D": [], "US_C": [], "AS_D": [], "AS_C": [], "AI": [], "AV": [], "UV": [], "AR": [], "UR": []}

    # 执行模拟
    for step in range(30):
        densities = epidemic_model_tau.count_all()

        # 更新状态密度
        results["US_D"].append(densities["US_D"])
        results["US_C"].append(densities["US_C"])
        results["AS_D"].append(densities["AS_D"])
        results["AS_C"].append(densities["AS_C"])
        results["AI"].append(densities["AI"])
        results["AV"].append(densities["AV"])
        results["UV"].append(densities["UV"])
        results["AR"].append(densities["AR"])
        results["UR"].append(densities["UR"])

        epidemic_model_tau.MC_Simulation()

    return results


# 定义tau值和要进行模拟的次数
tau_values = [0.0]
num_simulations = 100

# 创建存储文件夹
folder_name = "Again_fig1"
os.makedirs(folder_name, exist_ok=True)

# 执行模拟并保存结果
for tau in tau_values:
    # 为每个状态初始化一个空DataFrame，用于存储所有模拟的结果
    dfs = {state: pd.DataFrame() for state in ["US_D", "US_C", "AS_D", "AS_C", "AI", "AV", "UV", "AR", "UR"]}
    # 创建上层和下层网络
    lower_net = nx.barabasi_albert_graph(2000,4)
    upper_net = lower_net

    # 添加进度条来跟踪模拟的进度
    with tqdm(total=num_simulations, desc=f"Simulating for tau={tau}") as pbar:
        for sim in range(num_simulations):
            results = run_simulation(tau,lower_net,upper_net)
            # 模拟进度条更新
            pbar.update(1)
            # 将此次模拟结果存入相应的DataFrame
            for state, values in results.items():
                dfs[state][sim] = values

    # 保存每个状态的DataFrame为CSV文件
    for state, df in dfs.items():
        df.to_csv(f"{folder_name}/tau={tau}+{state}.csv", index=False)

