import os
import Epidemic
import Game
import networkx as nx
import numpy as np
import copy
import pandas as pd
import scipy.sparse as sp
import math
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path


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


def draw_network_snapshot(G, strategy_states, t):
    """
    绘制网络快照。

    :param G: networkx图，网络状态。
    :param strategy_states: dict，节点策略状态，'C'为合作者，'D'为背叛者。
    :param t: int，当前时间点。
    """
    pos = nx.spring_layout(G, seed=42)  # 为了一致性，使用固定的种子
    cooperators = [node for node, strategy in strategy_states.items() if strategy == 'C']
    defectors = [node for node, strategy in strategy_states.items() if strategy == 'D']

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, nodelist=cooperators, node_color='yellow', node_shape='s', label='Cooperators',node_size=50)
    nx.draw_networkx_nodes(G, pos, nodelist=defectors, node_color='purple', node_shape='s', label='Defectors',node_size=50)
    plt.axis('off')
    plt.title(f"t={t}",fontsize=20)
    plt.savefig(f"{folder_name}/network_snapshot_t{t}.png",dpi=600)  # 保存到先前创建的文件夹中
    plt.close()

def Game_simulation_HSIRA(lower_net, upper_net,eff,tau):
    results = []  # 用于保存每次的结果
    lower_net = lower_net
    upper_net = upper_net
    # 创建博弈实例
    Game_sim = Game.Game(cost_v= 0.8,lower_net=lower_net,upper_net=upper_net,
                         alpha=0.5,
                         alpha_triangle=1.0,
                         delta=0.5,
                         delta_triangle=1.0,
                         tau=tau,
                         beta=0.4,
                         eff=eff,
                         omega=0.1,
                         omega_triangle=0.0,
                         eta=0.6,
                         gamma=0.3333
                         )

    for t in range(global_t+1):
        # if t in [1, 10, 100]:
        #     strategy_states = Game_sim.get_strategy_states()  # 假设这个方法返回节点策略状态
        #     draw_network_snapshot(Game_sim.lower_net, strategy_states, t)  # 绘制快照
        if t == 0:
            Game_sim.init_awareness(init_u=0.5)  # 上层意识初始化

            Game_sim.init_strategy(init_c=0.1)  # 节点策略初始化

            Game_sim.init_state()  # 下层状态初始化
            Game_sim.init_infect(init_i=0.05) # 下层感染初始化

            Game_sim.epidemic_mc(times = 30) # 传播

            Game_sim.compute_payoff() # 计算收益

            Game_sim.update_strategy_HSIRA(k=0.1) #策略选择

            print(Game_sim.vc()) # 网络中合作者密度，策略都确定下来了

            results.append(Game_sim.vc())  # 将结果保存到列表中
        else:

            Game_sim.init_state()  # 下层状态初始化

            Game_sim.init_infect(init_i=0.05)  # 下层感染初始化

            Game_sim.epidemic_mc(times=30) # 传播

            Game_sim.compute_payoff() # 计算收益

            Game_sim.update_strategy_HSIRA(k=0.1) #策略选择

            print(Game_sim.vc()) # 网络中合作者密度，策略都确定下来了
            results.append(Game_sim.vc())  # 将结果保存到列表中

    return results


if __name__ == "__main__":
    # 定义超参数
    global_t = 100
    lower_net = netsimplicial_random(2000, 2)
    upper_net = add_random_edges(lower_net, 1000)

    # 创建存储文件夹
    folder_name = "fig5"
    os.makedirs(folder_name, exist_ok=True)

    # 初始化cost_v的值
    eff_values = [i / 50.0 for i in range(1, 51)]

    # 遍历不同的tau值
    for tau in [0.1, 0.5, 1.0]:
        # 为每个tau值定义一个CSV文件，文件名包含tau的值
        csv_file_path = os.path.join(folder_name, f"simulation_results_tau_{tau}.csv")

        # 打开CSV文件准备写入
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # 遍历不同的cost_v值
            for eff in eff_values:
                # 运行游戏模拟并获取结果
                simulation_results = Game_simulation_HSIRA(lower_net, upper_net, eff, tau)

                # 将模拟结果作为一行写入CSV文件
                # 假设simulation_results是一个时间序列列表
                writer.writerow(simulation_results)

