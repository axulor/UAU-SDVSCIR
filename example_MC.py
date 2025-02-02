import os
import Epidemic
import Game
import networkx as nx
import numpy as np
import copy
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path

# 创建项目文件夹
current_file_path = Path(__file__).parent
relative_path = Path("project_3_MC")
absolute_path = current_file_path / relative_path
os.makedirs(absolute_path, exist_ok=True)

# 定义超参数
global_t = 100

def save_results_to_file(data, filename):
    '''
    保存计算的数据
    '''
    df = pd.DataFrame(data)
    if os.path.isfile(filename):
        df_old = pd.read_csv(filename, header=None)
        df_new = pd.concat([df_old, df], axis=1, ignore_index=True)
    else:
        df_new = df
    df_new.to_csv(filename, index=False, header=False)

def create_network_graph(net):
    '''
    根据网络状态创建一个网络图。
    '''
    infected_color = '#9F0A09'  # 红色
    freeloader_color = '#F7DA43'  # 黄色
    cooperator_color = '#21A28E'  # 绿色

    # 为每个节点分配一个颜色
    colors = []
    for node, data in net.nodes(data=True):
        if data['state'] in ['I', 'R']:
            colors.append(infected_color)
        elif data['state'] == 'S_D':
            colors.append(freeloader_color)
        elif data['state'] in ['V', 'S_C']:
            colors.append(cooperator_color)

    # 绘制网络图
    nx.draw(net, node_color=colors, edge_color='lightgray')

def MC_simulation(seed):
    results = []  # 用于保存每次的结果
    lower_net = nx.barabasi_albert_graph(1000, 5, seed = int(seed * np.random.rand() * 100 ))
    upper_net = Game.add_random_edges(lower_net, 200)

    # 创建博弈实例
    Game_sim = Game.Game(cost_v= 0.4,lower_net=lower_net,upper_net=upper_net,
                         alpha = 0.6, delta = 0.4, beta = 0.4, eff = 0.7, omega = 0.1, eta = 0.6, gamma = 0.3333)

    for t in range(global_t):
        if t == 0:
            Game_sim.init_awareness(init_u=0.99)  # 上层意识初始化

            Game_sim.init_strategy(init_c=0.1)  # 节点策略初始化

            Game_sim.init_state()  # 下层状态初始化
            Game_sim.init_infect(init_i=0.02) # 下层感染初始化

            Game_sim.epidemic_mc(times = 30) # 传播

            Game_sim.compute_payoff() # 计算收益

            Game_sim.update_strategy(k=0.1) #策略选择

            print(Game_sim.vc()) # 网络中合作者密度，策略都确定下来了

            results.append(Game_sim.vc())  # 将结果保存到列表中
        else:
            # Game_sim.set_omega(omega=0.2)

            Game_sim.init_state()  # 下层状态初始化

            Game_sim.init_infect(init_i=0.02)  # 下层感染初始化

            Game_sim.epidemic_mc(times=30) # 传播

            Game_sim.compute_payoff() # 计算收益

            Game_sim.update_strategy(k=0.1) #策略选择

            print(Game_sim.vc()) # 网络中合作者密度，策略都确定下来了
            results.append(Game_sim.vc())  # 将结果保存到列表中



        # # here, take snapshots at t = 1, 10, and 100
        # if t in [0, 9, 99]:
        #     plt.figure(figsize=(10, 10))
        #     create_network_graph(Game_sim.lower_net)
        #     plt.axis('off')  # 去掉坐标轴
        #     plt.savefig(absolute_path / f"cost_v{Game_sim.cost_v}-eff{Game_sim.eff}-t{t}.svg", bbox_inches='tight', pad_inches=0)

    save_results_to_file(results, absolute_path / f"MC_t_cost_v{Game_sim.cost_v}-eff{Game_sim.eff}.csv")  # 将结果保存到文件中
    return np.mean(results[-60:])


if __name__ == "__main__":
    avg_last_30 = MC_simulation(1)
    print("Average of the last 30 values: ", avg_last_30)
