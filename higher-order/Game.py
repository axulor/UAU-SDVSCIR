# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2023-02-06

import networkx as nx
import numpy as np
import math
import random
import copy
from Epidemic import Epidemic


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

    def update_strategy_IBRA(self, k):
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

    def update_strategy_HSIRA(self, k):
        for node in self.lower_net.nodes():
            # 将下一步策略默认为当前策略
            self.lower_net.nodes[node]["next_strategy"] = self.lower_net.nodes[node]["strategy"]
            triangles = self.find_triangles(node)
            if not triangles:
                continue
            # 随机选中一个三角形
            selected_triangle = random.choice(triangles)

            # 从选中的三角形中排除当前节点，仅考虑其他两个节点
            triangle_neighbors = [n for n in selected_triangle if n != node]

            # 计算除当前节点以外的另外两个节点的平均收益和模仿概率
            selected_avg_payoff = sum(self.lower_net.nodes[n]["payoff"] for n in triangle_neighbors) / 2
            imitation_probability = 1 / (
                    1 + np.exp((self.lower_net.nodes[node]["payoff"] - selected_avg_payoff) / k))

            # 随机选取三角形中的邻居节点
            selected_neighbor = random.choice(triangle_neighbors)
            # 以概率模仿
            if random.random() < imitation_probability * self.tau:
                imitated_strategy = self.lower_net.nodes[selected_neighbor]["strategy"]
            else:
                imitated_strategy = self.lower_net.nodes[node]["strategy"]

            self.lower_net.nodes[node]["next_strategy"] = imitated_strategy

        # 策略更新
        for node in self.lower_net.nodes():
            self.lower_net.nodes[node]["strategy"] = self.lower_net.nodes[node]["next_strategy"]

    def find_triangles(self, node):
        # 初始化三角形列表
        triangles = []
        # 获取指定节点的所有邻居
        neighbors = list(self.lower_net.neighbors(node))
        # 遍历每个邻居，寻找共同邻居以形成三角形
        for i, neighbor in enumerate(neighbors):
            # 遍历当前邻居之后的邻居，避免重复检查并确保三角形的唯一性
            for second_neighbor in neighbors[i + 1:]:
                # 如果当前邻居和另一个邻居之间存在边，则形成了一个三角形
                if self.lower_net.has_edge(neighbor, second_neighbor):
                    # 形成的三角形
                    triangle = sorted([node, neighbor, second_neighbor])
                    # 添加到三角形列表中，由于使用了排序和集合，每个三角形只会被记录一次
                    if triangle not in triangles:
                        triangles.append(triangle)
        return triangles

    def show(self):
        for node in self.upper_net:
            print(node, self.upper_net.nodes[node],self.lower_net.nodes[node])

    def get_strategy_states(self):
        """
        获取所有节点的策略状态。

        :return: 一个字典，键是节点的标识符，值是策略状态（"C"表示合作者，"D"表示背叛者）。
        """
        strategy_states = {}
        for node in self.lower_net.nodes:
            strategy = self.lower_net.nodes[node].get("strategy", None)  # 获取节点的策略，如果没有则返回None
            strategy_states[node] = strategy

        return strategy_states

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




