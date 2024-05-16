# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2023-02-06

import networkx as nx
import numpy as np
import math
import random
import copy
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


class Game(Epidemic):

    def __init__(self, cost_v, **kwargs):
        '''
        :param cost_v: 疫苗相对接种成本
        :param kwargs: 父类Epidemic的所有参数
        '''
        super().__init__(**kwargs)
        self.cost_v = cost_v

    def epidemic_mc(self,times):

        # 对网络状态进行蒙特卡洛模拟更新
        for t in range(0, times):
            self.MC_Simulation()

        # 记录稳态为终止状态, I,R为 Infected, S,V为 Healthy
        for node in self.lower_net:
            end_state = self.lower_net.nodes[node].setdefault("end_state", None)
            if end_state is None:
                if self.lower_net.nodes[node]["state"] == "I" or self.lower_net.nodes[node]["state"] == "R":
                    self.lower_net.nodes[node]["end_state"] = "Infected"
                else:
                    self.lower_net.nodes[node]["end_state"] = "Healthy"

    def compute_payoff(self):
        '''
        对网络中所有节点计算收益
        '''
        # 定义收益字典
        payoff_dict = {
            ("C", "Healthy"): -self.cost_v,
            ("C", "Infected"): -self.cost_v - 1,
            ("D", "Healthy"): 0,
            ("D", "Infected"): -1,
        }
        # 计算每个节点的收益
        for node in self.lower_net:
            strategy = self.lower_net.nodes[node]["strategy"]
            end_state = self.lower_net.nodes[node]["end_state"]
            self.lower_net.nodes[node]["payoff"] = payoff_dict[(strategy, end_state)]

    def update_strategy(self, k):
        for node in self.lower_net:
            # 随机选择一个邻居来模仿
            neighbors = list(self.upper_net.adj[node])
            imitated_neighbor = random.choice(neighbors)
            # 比较收益，以概率模仿邻居
            imitation_probability = 1 / \
                                    (1 + np.exp((self.lower_net.nodes[node]["payoff"]
                                                 - self.lower_net.nodes[imitated_neighbor]["payoff"]) / k))
            if random.random() < imitation_probability:
                self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[imitated_neighbor]["strategy"]
            else:
                self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[node]["strategy"]
        # 策略更新
        for node in self.lower_net:
            self.lower_net.nodes[node]["strategy"] = self.lower_net.nodes[node]["next_strategy"]

    # def update_strategy(self, k):
    #     for node in self.lower_net:
    #         neighbors = list(self.upper_net.adj[node])
    #         probabilities = []  # List to store the probability of imitating each neighbor
    #
    #         for imitated_neighbor in neighbors:
    #             # Calculate the probability of imitating this neighbor
    #             imitation_probability = 1 / \
    #                                     (1 + np.exp((self.lower_net.nodes[node]["payoff"]
    #                                                  - self.lower_net.nodes[imitated_neighbor]["payoff"]) / k))
    #             probabilities.append(imitation_probability)
    #
    #         # Normalize the probabilities so that they sum to 1
    #         probabilities = [p / sum(probabilities) for p in probabilities]
    #
    #         # Choose a neighbor to imitate based on the calculated probabilities
    #         imitated_neighbor = np.random.choice(neighbors, p=probabilities)
    #
    #         # Imitate the chosen neighbor's strategy
    #         self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[imitated_neighbor]["strategy"]
    #
    #     # Update the strategies
    #     for node in self.lower_net:
    #         self.lower_net.nodes[node]["strategy"] = self.lower_net.nodes[node]["next_strategy"]

    def update_strategy_SBRA(self, k):
        '''
        基于策略的更新方式
        '''
        c_payoff = 0
        c_num = 1 # 防止除零错误
        d_payoff = 0
        d_num = 1 # 防止除零错误
        for node in self.lower_net:
            # 计算C,D两种策略的平均收益
            if self.lower_net.nodes[node]["strategy"] == "C":
                c_num += 1
                c_payoff += self.lower_net.nodes[node]["payoff"]
            elif self.lower_net.nodes[node]["strategy"] == "D":
                d_num += 1
                d_payoff += self.lower_net.nodes[node]["payoff"]
        c_avg_payoff = c_payoff / c_num
        d_avg_payoff = d_payoff / d_num

        for node in self.lower_net:

            # # 先更新自身策略的平均收益
            # if self.lower_net.nodes[node]["strategy"] == 'C':
            #     self.lower_net.nodes[node]["payoff"]  = c_avg_payoff
            # elif self.lower_net.nodes[node]["strategy"] == 'D':
            #     self.lower_net.nodes[node]["payoff"]  = d_avg_payoff

            # 随机选择一个邻居来模仿
            neighbors = list(self.upper_net.adj[node])
            imitated_neighbor = random.choice(neighbors)
            # 先更新邻居策略的平均收益
            if self.lower_net.nodes[imitated_neighbor]["strategy"] == 'C':
                self.lower_net.nodes[imitated_neighbor]["payoff"]  = c_avg_payoff
            elif self.lower_net.nodes[imitated_neighbor]["strategy"] == 'D':
                self.lower_net.nodes[imitated_neighbor]["payoff"]  = d_avg_payoff
            # 比较收益，以概率模仿邻居
            imitation_probability = 1 / \
                                        (1 + np.exp((self.lower_net.nodes[node]["payoff"]
                                                     - self.lower_net.nodes[imitated_neighbor]["payoff"]) / k))
            if random.random() < imitation_probability:
                self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[imitated_neighbor]["strategy"]
            else:
                self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[node]["strategy"]

        # 策略更新
        for node in self.lower_net:
            self.lower_net.nodes[node]["strategy"] = self.lower_net.nodes[node]["next_strategy"]

    def update_strategy_NA(self, k,proportion=0.8 ):
        '''
        不完全的更新方式
        '''
        all_nodes = list(self.lower_net.nodes)
        num_update = int(len(all_nodes) * proportion)
        update_nodes = random.sample(all_nodes, num_update)

        # 统计策略"D"和"C"的数量
        num_D = sum([1 for node in self.lower_net.nodes if self.lower_net.nodes[node]["strategy"] == "D"])
        num_C = sum([1 for node in self.lower_net.nodes if self.lower_net.nodes[node]["strategy"] == "C"])

        # 确定占比较小的策略
        minority_strategy = "D" if num_D < num_C else "C"

        for node in self.lower_net:
            if node in update_nodes:
                # 随机选择一个邻居来模仿
                neighbors = list(self.upper_net.adj[node])
                imitated_neighbor = random.choice(neighbors)

                imitation_probability = 1 / \
                                        (1 + np.exp((self.lower_net.nodes[node]["payoff"]
                                                    - self.lower_net.nodes[imitated_neighbor]["payoff"]) / k))
                if random.random() < imitation_probability:
                    self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[imitated_neighbor]["strategy"]
                else:
                    self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[node]["strategy"]
            else:
                self.lower_net.nodes[node]["next_strategy"] = minority_strategy
        # 策略更新
        for node in self.lower_net:
            self.lower_net.nodes[node]["strategy"] = self.lower_net.nodes[node]["next_strategy"]

    def show(self):
        for node in self.upper_net:
            print(node, self.upper_net.nodes[node],self.lower_net.nodes[node])

    def set_omega(self, omega=None):
        if omega is not None:
            self.omega = omega

    def fes(self):
        '''
        计算最终流行病规模
        :return: fes 疾病最终流行规模
        '''
        num_r = 0
        for node in self.lower_net:
            if self.lower_net.nodes[node]["state"] == "R":
                num_r += 1
        fes = num_r / len(self.lower_net)
        return fes

    def vc(self):
        '''
        计算人群合作水平
        :return: c 人群合作密度
        '''
        num_cooperator = 0
        for node in self.lower_net:
            if self.lower_net.nodes[node]["strategy"] == "C":
                num_cooperator += 1
        c = num_cooperator / len(self.lower_net)
        return c

    def asp(self):
        '''
        计算社会平均成本
        :return: c 人群合作密度
        '''
        sp = 0
        for node in self.lower_net:
            sp += self.lower_net.nodes[node]["payoff"]

        return sp / len(self.lower_net)


