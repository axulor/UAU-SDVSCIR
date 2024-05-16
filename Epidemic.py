# -*- coding:utf-8 -*-
# Author：wangyc
# CreateTime：2023-02-06

import numpy as np
import pandas as pd
import networkx as nx
import random
import concurrent.futures
import copy


class Epidemic:
    def __init__(self, lower_net, upper_net, alpha, delta, beta, eff, omega, eta, gamma):
        '''
        :param lower_net: 下层网络，疾病传播层
        :param upper_net: 上层网络，意识传播层
        :param alpha: 无意识受影响变为有意识的转化率
        :param delta: 有意识的遗忘率
        :param beta: 疾病感染率
        :param eff: 疫苗有效率
        :param omega: 主动接种概率
        :param eta: 防护措施有效率
        :param gamma: 疾病恢复率
        '''
        self.lower_net = lower_net
        self.upper_net = upper_net
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.eff = eff
        self.omega = omega
        self.eta = eta
        self.gamma = gamma

    def init_awareness(self, init_u):
        '''
        对人群意识进行初始化
        :param init_u: 初始无意识人群的比率
        '''
        num_init_u = int(init_u * len(self.upper_net.nodes()))
        init_u = random.sample(self.upper_net.nodes(), num_init_u)
        for node in self.upper_net:
            if node in init_u:
                self.upper_net.nodes[node]["awareness"] = "U"
            else:
                self.upper_net.nodes[node]["awareness"] = "A"

    def init_strategy(self, init_c):
        '''
        对人群策略进行初始化
        :param init_c: 初始合作者比率
        :return:
        '''
        num_init_c = int(len(self.lower_net.nodes()) * init_c)
        init_c = random.sample(self.upper_net.nodes(), num_init_c)

        for node in self.lower_net:
            if node in init_c:
                self.lower_net.nodes[node]["strategy"] = "C"
            else:
                self.lower_net.nodes[node]["strategy"] = "D"

    def init_state(self):
        '''
        根据人群策略初始化状态,同时判断疫苗接种的效果
        '''
        for node in self.lower_net:
            if self.lower_net.nodes[node]["strategy"] == "C":
                self.lower_net.nodes[node]["state"] = "S_C"

                # 生成一个随机数，若这个随机数小于疫苗有效率，则疫苗生效
                if random.random() < self.eff:
                    self.lower_net.nodes[node]["state"] = "V"
                    self.lower_net.nodes[node]["end_state"] = "Healthy"
            else:
                self.lower_net.nodes[node]["state"] = "S_D"

    def init_infect(self, init_i):
        '''
        感染初始化
        :param init_i: 初始感染率
        '''
        num_init_i = int(init_i * len(self.lower_net.nodes()))
        non_v_nodes = [node for node in self.lower_net if self.lower_net.nodes[node]["state"] != "V"]

        # 如果要初始化的感染者数量大于非V状态的节点数量，则将所有非V状态的节点都设置为感染者
        if num_init_i >= len(non_v_nodes):
            init_i_nodes = non_v_nodes
        else:
            init_i_nodes = random.sample(non_v_nodes, num_init_i)

        for node in init_i_nodes:
            self.lower_net.nodes[node]["state"] = "I"
            self.lower_net.nodes[node]["end_state"] = "Infected"
            if self.upper_net.nodes[node]["awareness"] == "U":
                self.upper_net.nodes[node]["awareness"] = "A"

    def count_all(self):
        '''
        返回人群中的全部状态密度元组
        '''
        counters = {"US_D": 0, "AS_D": 0, "US_C": 0, "AS_C": 0, "AI": 0, "UR": 0, "AR": 0, "UV": 0, "AV": 0}
        num_nodes = len(self.lower_net.nodes)

        for node in self.lower_net:
            upper_net_node = self.upper_net.nodes[node]
            lower_net_node = self.lower_net.nodes[node]
            awareness = upper_net_node["awareness"]
            state = lower_net_node["state"]

            if awareness == "U" and state == "S_D":
                counters["US_D"] += 1
            elif awareness == "A" and state == "S_D":
                counters["AS_D"] += 1
            elif awareness == "U" and state == "S_C":
                counters["US_C"] += 1
            elif awareness == "A" and state == "S_C":
                counters["AS_C"] += 1
            elif state == "I":
                counters["AI"] += 1
            elif awareness == "U" and state == "R":
                counters["UR"] += 1
            elif awareness == "A" and state == "R":
                counters["AR"] += 1
            elif awareness == "U" and state == "V":
                counters["UV"] += 1
            elif awareness == "A" and state == "V":
                counters["AV"] += 1

        return tuple([value / num_nodes for value in counters.values()])

    def count_density(self, awareness, state):
        '''
        返回人群中的指定状态的密度
        '''
        num_nodes = 0
        for node in self.lower_net:
            if self.upper_net.nodes[node]["awareness"] == awareness and self.lower_net.nodes[node]["state"] == state:
                num_nodes += 1
        return num_nodes / len(self.lower_net)


    def MMCA(self, T, init_u, init_i, init_c=None, updated_p_C=None):
        '''
        对双层网络上的传播进行MMCA理论计算,考虑了群体免疫效应
        :param T: 总时间步
        :param U_Net: 上层网络邻接矩阵
        :param L_Net: 下层网络邻接矩阵
        :param alpha: 意识传播率
        :param delta: 意识遗忘率
        :param beta: 疾病传播率
        :param eff: 疫苗有效率
        :param omega: 警惕系数
        :param eta: 防护措施有效率
        :param gamma: 疾病恢复率
        :return: 七种状态的 T * N 矩阵（T行表示时间步，N列表示节点）
        '''

        U_Net = nx.to_numpy_array(self.upper_net)
        L_Net = nx.to_numpy_array(self.lower_net)
        r, k, q_u, q_a = [np.zeros((T, len(U_Net))) for i in range(4)]
        US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV = [np.zeros((T, len(U_Net))) for i in range(9)]

        for j in range(len(U_Net)):
            # Use updated_p_C if provided, otherwise use init_c
            c_value = updated_p_C[j] if updated_p_C else init_c

            US_D[0, j] = init_u * (1 - c_value) * (1 - init_i)
            AS_D[0, j] = (1 - init_u) * (1 - c_value) * (1 - init_i)
            US_C[0, j] = init_u * c_value * (1 - self.eff) * (1 - init_i)
            AS_C[0, j] = (1 - init_u) * c_value * (1 - self.eff) * (1 - init_i)
            AI[0, j] = init_i
            UR[0, j] = 0
            AR[0, j] = 0
            UV[0, j] = init_u * c_value * self.eff * (1 - init_i)
            AV[0, j] = (1 - init_u) * c_value * self.eff * (1 - init_i)

        for t in range(T - 1):
            for i in range(len(U_Net)):
                rr = kk = qu = qa = 1

                for j in range(len(L_Net)):
                    if U_Net[j, i] !=0 :
                        # rr: U状态没有被A状态邻居影响的概率, kk: S_D状态没有被感染状态影响到的概率
                        rr *= (1 - (AS_D[t, j] + AS_C[t, j] + AI[t, j] + AR[t, j] + AV[t, j]) * self.alpha )
                        kk *= (1 - AI[t, j] * self.omega * (1- UV[t,j] - AV[t,j]) ) #(1+ UV[t,j] + AV[t,j]+ UR[t,j]+ AR[t,j]))
                    if L_Net[j, i] != 0:
                        # qu: U态没有被感染的概率 , qa: A态没有被感染的概率
                        qu *= (1 - AI[t, j] * self.beta * (1- UV[t,j] - AV[t,j]) ) # - AV[t,j]
                        qa *= (1 - AI[t, j] * self.beta * (1 - self.eta) * (1- UV[t,j] - AV[t,j]) )  # - AV[t,j]

                r[t, i] = rr
                k[t, i] = kk
                q_u[t, i] = qu
                q_a[t, i] = qa

                # MMCA方程
                US_D[t + 1, i] = US_D[t, i] * r[t, i] * q_u[t, i] + AS_D[t, i] * self.delta * q_u[t, i]

                AS_D[t + 1, i] = US_D[t, i] * (1 - r[t, i]) * k[t, i] * q_a[t, i] \
                                 + AS_D[t, i] * (1 - self.delta) * k[t, i] * q_a[t, i]

                US_C[t + 1, i] = US_C[t, i] * r[t, i] * q_u[t, i] + AS_C[t, i] * self.delta * q_u[t, i]

                AS_C[t + 1, i] = US_D[t, i] * (1 - r[t, i]) * (1 - k[t, i]) * (1 - self.eff) * q_a[t, i] \
                                 + AS_D[t, i] * (1 - self.delta) * (1 - k[t, i]) * (1 - self.eff) * q_a[t, i] \
                                 + US_C[t, i] * (1 - r[t, i]) * q_a[t, i] \
                                 + AS_C[t, i] * (1 - self.delta) * q_a[t, i]

                AI[t + 1, i] = US_D[t, i] * r[t, i] * (1 - q_u[t, i]) + US_D[t, i] * (1 - r[t, i]) * k[t, i] * (
                        1 - q_a[t, i]) \
                               + US_D[t, i] * (1 - r[t, i]) * (1 - k[t, i]) * (1 - self.eff) * (1 - q_a[t, i]) \
                               + AS_D[t, i] * self.delta * (1 - q_u[t, i]) + AS_D[t, i] * (1 - self.delta) * k[t, i] * (
                                       1 - q_a[t, i]) \
                               + AS_D[t, i] * (1 - self.delta) * (1 - k[t, i]) * (1 - self.eff) * (1 - q_a[t, i]) \
                               + US_C[t, i] * r[t, i] * (1 - q_u[t, i]) + US_C[t, i] * (1 - r[t, i]) * (1 - q_a[t, i]) \
                               + AS_C[t, i] * self.delta * (1 - q_u[t, i]) + AS_C[t, i] * (1 - self.delta) * (
                                       1 - q_a[t, i]) \
                               + AI[t, i] * (1 - self.gamma)

                UR[t + 1, i] = AR[t, i] * self.delta + UR[t, i] * r[t, i]

                AR[t + 1, i] = AI[t, i] * self.gamma + AR[t, i] * (1 - self.delta) + UR[t, i] * (1 - r[t, i])

                UV[t + 1, i] = UV[t, i] * r[t, i] + AV[t, i] * self.delta

                AV[t + 1, i] = US_D[t, i] * (1 - r[t, i]) * (1 - k[t, i]) * self.eff + AS_D[t, i] * (1 - self.delta) * (
                        1 - k[t, i]) * self.eff \
                               + UV[t, i] * (1 - r[t, i]) + AV[t, i] * (1 - self.delta)

        return US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV

    def MMCAR(self, T, init_u, init_i, init_c=None, updated_p_C=None):
        '''
        对双层网络上的传播进行理论计算，未考虑群体免疫效应
        :param T: 总时间步
        :param U_Net: 上层网络邻接矩阵
        :param L_Net: 下层网络邻接矩阵
        :param alpha: 意识传播率
        :param delta: 意识遗忘率
        :param beta: 疾病传播率
        :param eff: 疫苗有效率
        :param omega: 警惕系数
        :param eta: 防护措施有效率
        :param gamma: 疾病恢复率
        :return: 七种状态的 T * N 矩阵（T行表示时间步，N列表示节点）
        '''

        U_Net = nx.to_numpy_array(self.upper_net)
        L_Net = nx.to_numpy_array(self.lower_net)
        r, k, q_u, q_a = [np.zeros((T, len(U_Net))) for i in range(4)]
        US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV = [np.zeros((T, len(U_Net))) for i in range(9)]

        for j in range(len(U_Net)):
            # Use updated_p_C if provided, otherwise use init_c
            c_value = updated_p_C[j] if updated_p_C else init_c

            US_D[0, j] = init_u * (1 - c_value) * (1 - init_i)
            AS_D[0, j] = (1 - init_u) * (1 - c_value) * (1 - init_i)
            US_C[0, j] = init_u * c_value * (1 - self.eff) * (1 - init_i)
            AS_C[0, j] = (1 - init_u) * c_value * (1 - self.eff) * (1 - init_i)
            AI[0, j] = init_i
            UR[0, j] = 0
            AR[0, j] = 0
            UV[0, j] = init_u * c_value * self.eff * (1 - init_i)
            AV[0, j] = (1 - init_u) * c_value * self.eff * (1 - init_i)

        for t in range(T - 1):
            for i in range(len(U_Net)):
                rr = kk = qu = qa = 1

                for j in range(len(L_Net)):
                    if U_Net[j, i] != 0:
                        # rr: U状态没有被A状态邻居影响的概率, kk: S_D状态没有被感染状态影响到的概率
                        rr *= (1 - (AS_D[t, j] + AS_C[t, j] + AI[t, j] + AR[t, j] + AV[t, j]) * self.alpha)
                        kk *= (1 - AI[t, j] * self.omega )  # (1+ UV[t,j] + AV[t,j]+ UR[t,j]+ AR[t,j]))
                    if L_Net[j, i] != 0:
                        # qu: U态没有被感染的概率 , qa: A态没有被感染的概率
                        qu *= (1 - AI[t, j] * self.beta )  # - AV[t,j]
                        qa *= (1 - AI[t, j] * self.beta * (1 - self.eta) )  # - AV[t,j]

                r[t, i] = rr
                k[t, i] = kk
                q_u[t, i] = qu
                q_a[t, i] = qa

                # MMCA方程
                US_D[t + 1, i] = US_D[t, i] * r[t, i] * q_u[t, i] + AS_D[t, i] * self.delta * q_u[t, i]

                AS_D[t + 1, i] = US_D[t, i] * (1 - r[t, i]) * k[t, i] * q_a[t, i] \
                                 + AS_D[t, i] * (1 - self.delta) * k[t, i] * q_a[t, i]

                US_C[t + 1, i] = US_C[t, i] * r[t, i] * q_u[t, i] + AS_C[t, i] * self.delta * q_u[t, i]

                AS_C[t + 1, i] = US_D[t, i] * (1 - r[t, i]) * (1 - k[t, i]) * (1 - self.eff) * q_a[t, i] \
                                 + AS_D[t, i] * (1 - self.delta) * (1 - k[t, i]) * (1 - self.eff) * q_a[t, i] \
                                 + US_C[t, i] * (1 - r[t, i]) * q_a[t, i] \
                                 + AS_C[t, i] * (1 - self.delta) * q_a[t, i]

                AI[t + 1, i] = US_D[t, i] * r[t, i] * (1 - q_u[t, i]) + US_D[t, i] * (1 - r[t, i]) * k[t, i] * (
                        1 - q_a[t, i]) \
                               + US_D[t, i] * (1 - r[t, i]) * (1 - k[t, i]) * (1 - self.eff) * (1 - q_a[t, i]) \
                               + AS_D[t, i] * self.delta * (1 - q_u[t, i]) + AS_D[t, i] * (1 - self.delta) * k[t, i] * (
                                       1 - q_a[t, i]) \
                               + AS_D[t, i] * (1 - self.delta) * (1 - k[t, i]) * (1 - self.eff) * (1 - q_a[t, i]) \
                               + US_C[t, i] * r[t, i] * (1 - q_u[t, i]) + US_C[t, i] * (1 - r[t, i]) * (1 - q_a[t, i]) \
                               + AS_C[t, i] * self.delta * (1 - q_u[t, i]) + AS_C[t, i] * (1 - self.delta) * (
                                       1 - q_a[t, i]) \
                               + AI[t, i] * (1 - self.gamma)

                UR[t + 1, i] = AR[t, i] * self.delta + UR[t, i] * r[t, i]

                AR[t + 1, i] = AI[t, i] * self.gamma + AR[t, i] * (1 - self.delta) + UR[t, i] * (1 - r[t, i])

                UV[t + 1, i] = UV[t, i] * r[t, i] + AV[t, i] * self.delta

                AV[t + 1, i] = US_D[t, i] * (1 - r[t, i]) * (1 - k[t, i]) * self.eff + AS_D[t, i] * (1 - self.delta) * (
                        1 - k[t, i]) * self.eff \
                               + UV[t, i] * (1 - r[t, i]) + AV[t, i] * (1 - self.delta)

        return US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV


    def MC_Simulation(self):
        '''
        对双层网络上的传播进行蒙特卡洛模拟
        :param self.upper_net: 上层网络，意识传播层
        :param self.lower_net: 下层网络，疾病传播层
        :param alpha: 无意识受邻居影响转变为有意识的概率
        :param delta: 有意识的遗忘率
        :param beta: 疾病传染率
        :param eff: 疫苗有效率
        :param omega: 主动接种率
        :param eta: 有意识防护有效率
        :param gamma: 疾病恢复率
        :return: 同步更新一次MC模拟之后的网络
        '''

        for node in self.lower_net:
            # US_D
            if self.upper_net.nodes[node]["awareness"] == "U" and self.lower_net.nodes[node]["state"] == "S_D":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "U"
                self.lower_net.nodes[node]["next_state"] = "S_D"
                # 意识传播，遍历所有上层邻居
                for neighbor in self.upper_net.adj[node]:
                    if self.upper_net.nodes[neighbor]["awareness"] == "A":  # 如果这个邻居有防护意识
                        p = np.random.rand()
                        if p < self.alpha:  # 被告知成为有防护意识
                            self.upper_net.nodes[node]["next_awareness"] = "A"
                            break
                # 疫苗接种，遍历所有上层邻居
                if self.upper_net.nodes[node]["next_awareness"] == "A":
                    for neighbor in self.upper_net.adj[node]:
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p1 = np.random.rand()
                            if p1 < self.omega:  # 采取接种策略
                                self.lower_net.nodes[node]["next_state"] = "S_C"
                                self.lower_net.nodes[node]["strategy"] = "C"  # 用于博弈中的是否参与接种的记录
                                p2 = np.random.rand()
                                if p2 < self.eff:  # 判断疫苗是否有效
                                    self.lower_net.nodes[node]["next_state"] = "V"
                                    self.lower_net.nodes[node]["end_state"] = "Healthy"
                                break  # 找到一个感染者邻居并采取接种策略后，跳出循环
                # 疾病传播，遍历所有下层邻居
                if self.upper_net.nodes[node]["next_awareness"] == "U" and (self.lower_net.nodes[node][
                    "next_state"] == "S_D" or self.lower_net.nodes[node]["next_state"] == "S_C"):
                    for neighbor in self.lower_net.adj[node]:
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p = np.random.rand()
                            if p < self.beta:  # 易感者被感染
                                self.lower_net.nodes[node]["next_state"] = "I"
                                self.upper_net.nodes[node]["next_awareness"] = "A"
                                self.lower_net.nodes[node]["end_state"] = "Infected"
                                break
                elif self.upper_net.nodes[node]["next_awareness"] == "A" and (self.lower_net.nodes[node][
                    "next_state"] == "S_D" or self.lower_net.nodes[node]["next_state"] == "S_C"):
                    for neighbor in self.lower_net.adj[node]:  # 遍历所有下层邻居
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p1 = np.random.rand()
                            if p1 < self.beta:  # 易感者被感染
                                p2 = np.random.rand()
                                if p2 < (1 - self.eta):  # 防护措施无效
                                    self.lower_net.nodes[node]["next_state"] = "I"
                                    self.lower_net.nodes[node]["end_state"] = "Infected"
                                    break
            # AS_D
            elif self.upper_net.nodes[node]["awareness"] == "A" and self.lower_net.nodes[node]["state"] == "S_D":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "A"
                self.lower_net.nodes[node]["next_state"] = "S_D"
                # 意识传播
                p = np.random.rand()
                if p < self.delta:  # 以概率遗忘
                    self.upper_net.nodes[node]["next_awareness"] = "U"
                # 疫苗接种，遍历所有上层邻居
                if self.upper_net.nodes[node]["next_awareness"] == "A":
                    for neighbor in self.upper_net.adj[node]:
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p1 = np.random.rand()
                            if p1 < self.omega:  # 采取接种策略
                                self.lower_net.nodes[node]["next_state"] = "S_C"
                                self.lower_net.nodes[node]["strategy"] = "C"  # 用于博弈中的是否参与接种的记录
                                p2 = np.random.rand()
                                if p2 < self.eff:  # 判断疫苗是否有效
                                    self.lower_net.nodes[node]["next_state"] = "V"
                                    self.lower_net.nodes[node]["end_state"] = "Healthy"
                                break  # 找到一个感染者邻居并采取接种策略后，跳出循环
                # 疾病传播，遍历所有下层邻居
                if self.upper_net.nodes[node]["next_awareness"] == "U" and (self.lower_net.nodes[node][
                    "next_state"] == "S_D" or self.lower_net.nodes[node]["next_state"] == "S_C"):
                    for neighbor in self.lower_net.adj[node]:
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p = np.random.rand()
                            if p < self.beta:  # 易感者被感染
                                self.lower_net.nodes[node]["next_state"] = "I"
                                self.upper_net.nodes[node]["next_awareness"] = "A"
                                self.lower_net.nodes[node]["end_state"] = "Infected"
                                break
                elif self.upper_net.nodes[node]["next_awareness"] == "A" and (self.lower_net.nodes[node][
                    "next_state"] == "S_D" or self.lower_net.nodes[node]["next_state"] == "S_C"):
                    for neighbor in self.lower_net.adj[node]:  # 遍历所有下层邻居
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p1 = np.random.rand()
                            if p1 < self.beta:  # 易感者被感染
                                p2 = np.random.rand()
                                if p2 < (1 - self.eta):  # 防护措施无效
                                    self.lower_net.nodes[node]["next_state"] = "I"
                                    self.lower_net.nodes[node]["end_state"] = "Infected"
                                    break
            # US_C
            elif self.upper_net.nodes[node]["awareness"] == "U" and self.lower_net.nodes[node]["state"] == "S_C":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "U"
                self.lower_net.nodes[node]["next_state"] = "S_C"
                # 意识传播
                for neighbor in self.upper_net.adj[node]:  # 遍历所有上层邻居
                    if self.upper_net.nodes[neighbor]["awareness"] == "A":  # 如果这个邻居有防护意识
                        p = np.random.rand()
                        if p < self.alpha:  # 被告知成为有防护意识
                            self.upper_net.nodes[node]["next_awareness"] = "A"
                            break
                # 疾病传播
                if self.upper_net.nodes[node]["next_awareness"] == "U" :
                    for neighbor in self.lower_net.adj[node]:  # 遍历所有下层邻居
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p = np.random.rand()
                            if p < self.beta :  # 易感者被感染
                                self.lower_net.nodes[node]["next_state"] = "I"
                                self.upper_net.nodes[node]["next_awareness"] = "A"
                                self.lower_net.nodes[node]["end_state"] = "Infected"
                                break
                elif self.upper_net.nodes[node]["next_awareness"] == "A" :
                    for neighbor in self.lower_net.adj[node]:  # 遍历所有下层邻居
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p = np.random.rand()
                            if p < (1 - self.eta) :  # 防护措施无效
                                p2 = np.random.rand()
                                if p2 < self.beta:  # 易感者被感染
                                    self.lower_net.nodes[node]["next_state"] = "I"
                                    self.lower_net.nodes[node]["end_state"] = "Infected"
                                    break
            # AS_C
            elif self.upper_net.nodes[node]["awareness"] == "A" and self.lower_net.nodes[node]["state"] == "S_C":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "A"
                self.lower_net.nodes[node]["next_state"] = "S_C"
                p = np.random.rand()
                # 意识传播
                if p < self.delta:  # 以概率遗忘
                    self.upper_net.nodes[node]["next_awareness"] = "U"
                # 疾病传播
                if self.upper_net.nodes[node]["next_awareness"] == "U" :
                    for neighbor in self.lower_net.adj[node]:  # 遍历所有下层邻居
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p = np.random.rand()
                            if p < self.beta :  # 易感者被感染
                                self.lower_net.nodes[node]["next_state"] = "I"
                                self.upper_net.nodes[node]["next_awareness"] = "A"
                                self.lower_net.nodes[node]["end_state"] = "Infected"
                                break
                elif self.upper_net.nodes[node]["next_awareness"] == "A" :
                    for neighbor in self.lower_net.adj[node]:  # 遍历所有下层邻居
                        if self.lower_net.nodes[neighbor]["state"] == "I":  # 如果这个邻居是感染者
                            p = np.random.rand()
                            if p < self.beta:  # 易感者被感染
                                p2 = np.random.rand()
                                if p2 < (1 - self.eta):  # 防护措施无效
                                    self.lower_net.nodes[node]["next_state"] = "I"
                                    self.lower_net.nodes[node]["end_state"] = "Infected"
                                    break
            # AI
            elif self.lower_net.nodes[node]["state"] == "I":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "A"
                self.lower_net.nodes[node]["next_state"] = "I"
                self.lower_net.nodes[node]["end_state"] = "Infected"
                p = np.random.rand()
                if p < self.gamma:  # 恢复健康
                    self.lower_net.nodes[node]["next_state"] = "R"
            # AR
            elif self.upper_net.nodes[node]["awareness"] == "A" and self.lower_net.nodes[node]["state"] == "R":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "A"
                self.lower_net.nodes[node]["next_state"] = "R"
                # 意识传播，以概率遗忘
                p = np.random.rand()
                if p < self.delta:
                    self.upper_net.nodes[node]["next_awareness"] = "U"
            # UR
            elif self.upper_net.nodes[node]["awareness"] == "U" and self.lower_net.nodes[node]["state"] == "R":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "U"
                self.lower_net.nodes[node]["next_state"] = "R"
                # 意识传播，遍历所有上层邻居
                for neighbor in self.upper_net.adj[node]:
                    if self.upper_net.nodes[neighbor]["awareness"] == "A":  # 如果这个邻居有防护意识
                        p = np.random.rand()
                        if p < self.alpha:  # 被告知成为有防护意识
                            self.upper_net.nodes[node]["next_awareness"] = "A"
                            break
            # AV
            elif self.upper_net.nodes[node]["awareness"] == "A" and self.lower_net.nodes[node]["state"] == "V":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "A"
                self.lower_net.nodes[node]["next_state"] = "V"
                self.lower_net.nodes[node]["end_state"] = "Healthy"
                # 意识传播，以概率遗忘
                p = np.random.rand()
                if p < self.delta:  # 以概率遗忘
                    self.upper_net.nodes[node]["next_awareness"] = "U"
            # UV
            elif self.upper_net.nodes[node]["awareness"] == "U" and self.lower_net.nodes[node]["state"] == "V":
                # 默认下一状态
                self.upper_net.nodes[node]["next_awareness"] = "U"
                self.lower_net.nodes[node]["next_state"] = "V"
                self.lower_net.nodes[node]["end_state"] = "Healthy"
                # 意识传播，遍历所有上层邻居
                for neighbor in self.upper_net.adj[node]:
                    if self.upper_net.nodes[neighbor]["awareness"] == "A":  # 如果这个邻居有防护意识
                        p = np.random.rand()
                        if p < self.alpha:  # 被告知成为有防护意识
                            self.upper_net.nodes[node]["next_awareness"] = "A"
                            break
            # 网络更新
        for node in self.lower_net:
            self.upper_net.nodes[node]["awareness"] = self.upper_net.nodes[node]["next_awareness"]
            self.lower_net.nodes[node]["state"] = self.lower_net.nodes[node]["next_state"]
            if self.upper_net.nodes[node]["awareness"] == "U" and self.lower_net.nodes[node]["state"] == "I":
                self.upper_net.nodes[node]["awareness"] = "A"

    # 计算疾病传播情况
    def count_state(self):
        s_d = 0
        s_c = 0
        i = 0
        r = 0
        v = 0
        num_nodes = len(self.lower_net.nodes)
        for node in self.lower_net:
            if self.lower_net.nodes[node]["state"] == "S_D":
                s_d = s_d + 1
            elif self.lower_net.nodes[node]["state"] == "S_C":
                s_c = s_c + 1
            elif self.lower_net.nodes[node]["state"] == "I":
                i = i + 1
            elif self.lower_net.nodes[node]["state"] == "R":
                r = r + 1
            elif self.lower_net.nodes[node]["state"] == "V":
                v = v + 1
        return s_d / num_nodes, s_c / num_nodes, i / num_nodes, r / num_nodes, v / num_nodes

    # 计算意识传播情况
    def count_ua(self):
        u = 0
        a = 0
        num_nodes = len(self.upper_net.nodes)
        for node in self.upper_net:
            if self.upper_net.nodes[node]["awareness"] == "U":
                u = u + 1
            elif self.upper_net.nodes[node]["awareness"] == "A":
                a = a + 1
        return u / num_nodes, a / num_nodes

    def set_params(self, beta=None):
        if beta is not None:
            self.beta = beta


