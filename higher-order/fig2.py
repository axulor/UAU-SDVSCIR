import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
import copy
import scipy.sparse as sp
import numpy as np
from Epidemic import Epidemic


def netsimplicial_random(N, ntri):
    '''
    生成一个随机单纯形网络，通过随机连接节点来形成新的三角形。

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


def netsimplicial_preferential(N, ntri):
    '''
    生成一个具有偏好连接的随机单纯形网络。在这个网络中，新节点的连接不是完全随机的，
    而是更倾向于连接到那些已经参与了更多三角形的节点，即具有更高局部聚类系数的节点。

    :param N: int
        网络中的总节点数。这决定了最终生成网络的规模。
    :param ntri: int
        每次添加新节点时，通过偏好连接选择现有的边来与新节点连接，形成的三角形数量。
        这个参数控制了网络中三角形的形成速度和密度。
    :return: networkx.Graph
        生成的网络，表示为一个NetworkX图对象。这个图对象包含了网络的所有节点和边，
        可以用于进一步的分析或可视化。
    '''
    mlinks = 2 * ntri
    N0 = mlinks + 1
    A = sp.dok_matrix((N, N), dtype=np.float32)
    A[0:N0, 0:N0] = 1
    A.setdiag(0)  # 移除自环

    # 计算每条边形成三角形的次数
    Dij = A.multiply(A @ A)
    TDij = sp.triu(Dij)  # 获取上三角矩阵

    for n in range(N0, N):
        i, j = TDij.nonzero()
        # 转换TDij为csr_matrix以支持下标访问
        TDij_csr = TDij.tocsr()
        weights = TDij_csr[i, j].A1  # 使用转换后的矩阵来获取权重
        weights_cumsum = np.cumsum(weights)
        weights_cumsum /= weights_cumsum[-1]  # 归一化累积和

        selected_edges = set()
        while len(selected_edges) < ntri:
            rand_vals = np.random.rand(ntri)
            edge_idxs = np.searchsorted(weights_cumsum, rand_vals)
            selected_edges.update(edge_idxs.tolist())

        # 更新邻接矩阵和三角形次数矩阵
        for idx in selected_edges:
            inode, jnode = i[idx], j[idx]
            A[n, inode] = A[inode, n] = 1
            A[n, jnode] = A[jnode, n] = 1
            Dij[n, inode] = Dij[inode, n] = 1
            Dij[n, jnode] = Dij[jnode, n] = 1
            Dij[inode, jnode] = Dij[jnode, inode] = Dij[inode, jnode] + 1
            TDij = sp.triu(Dij)  # 更新上三角矩阵

    G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    return G


# 为了执行模拟，首先我们需要定义一个函数来执行单次模拟并收集数据
def once_simulation(beta_value, lower_net, upper_net):
    # 创建新的流行病模型实例，使用指定的tau值
    epidemic_model_beta = Epidemic(
        lower_net=lower_net,
        upper_net=upper_net,
        alpha=0.5,
        alpha_triangle=1.0,
        delta=0.5,
        delta_triangle=1.0,
        tau=0.0,
        beta=beta_value, # 使用函数参数中的beta值
        eff=0.8,
        omega=0.1,
        omega_triangle=1.0,
        eta=0.6,
        gamma=0.3333
    )

    # 初始化网络状态
    epidemic_model_beta.init_awareness(init_u=0.5)
    epidemic_model_beta.init_strategy(init_c=0.1)
    epidemic_model_beta.init_state()
    epidemic_model_beta.init_infect(init_i=0.1)

    # 存储每个状态的密度
    results = {"R": [], "C": []}

    # 执行模拟
    for step in range(30):
        densities = epidemic_model_beta.count_all()

        # 更新状态密度

        results["R"].append(densities["AR"] + densities["UR"])
        results["C"].append(epidemic_model_beta.count_c())

        epidemic_model_beta.MC_Simulation()


    # 计算最后五步的平均密度
    avg_R = sum(results["R"][-5:]) / 5
    avg_C = sum(results["C"][-5:]) / 5

    return avg_R, avg_C


def run_simulation(lower_net, upper_net, network_name, num_simulations, beta_values):
    # 初始化两个空的DataFrame，行数为beta_values的长度，列数为模拟次数
    column_names = [f"Simulation {i+1}" for i in range(num_simulations)]
    df_R = pd.DataFrame(index=beta_values, columns=column_names)
    df_C = pd.DataFrame(index=beta_values, columns=column_names)

    # 使用enumerate在循环中同时获取beta值和其索引
    for beta_index, beta in enumerate(tqdm(beta_values, desc="Total progress", unit="beta")):
        R_values = []  # 存储当前beta下所有模拟的R结果
        C_values = []  # 存储当前beta下所有模拟的C结果

        # 在内层循环中添加进度条来跟踪每个beta值下模拟的进度
        for sim in tqdm(range(num_simulations), desc=f"Simulating for beta={beta}", leave=False):
            avg_R, avg_C = once_simulation(beta, lower_net, upper_net)
            R_values.append(avg_R)
            C_values.append(avg_C)

        # 将模拟结果填充到对应的DataFrame行中
        df_R.loc[beta, :] = R_values
        df_C.loc[beta, :] = C_values

    # 保存DataFrame到CSV文件，确保文件被保存在fig2文件夹中
    df_R.to_csv(f"{folder_name}/{network_name}_R.csv")
    df_C.to_csv(f"{folder_name}/{network_name}_C.csv")


# 创建存储文件夹
folder_name = "Again_fig2_SR"
os.makedirs(folder_name, exist_ok=True)
# 定义beta_values范围
# Correcting the sequence to remove the duplicated 0.1 and properly increment by 0.03 up to 0.99, then include 1.0
beta_values = [round(x * 0.001, 3) for x in range(1, 11)] + \
                        [round(x * 0.01, 2) for x in range(2, 11)] + \
                        [round(0.13 + x * 0.03, 2) for x in range(int((0.99 - 0.13) / 0.03) + 1)] + \
                        [1.00]

# 模拟参数
num_simulations = 10
net_types = [1]

for net_type in tqdm(net_types, desc="Generating networks and running simulations"):

    # 根据网络类型生成下层网络
    lower_net = nx.barabasi_albert_graph(2000,4)
    upper_net = netsimplicial_random(2000,2)

    network_name='SR'

    # 执行模拟
    run_simulation(lower_net, upper_net, network_name, num_simulations, beta_values)


