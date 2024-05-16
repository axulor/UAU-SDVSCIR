import os
import Epidemic
import Game
import networkx as nx
import numpy as np
import copy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def calculate_mmca_avg(epidemic_instance):

    US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV = epidemic_instance.MMCA(T=30,init_u=0.99,init_c=0.1,init_i=0.02)

    # 求每个矩阵每一行的平均值
    mmca_avg = np.column_stack([np.mean(matrix, axis=1) for matrix in [US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV]])

    return mmca_avg

def calculate_mmcar_avg(epidemic_instance):

    US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV = epidemic_instance.MMCAR(T=30,init_u=0.99,init_c=0.1,init_i=0.02)
    # 求每个矩阵每一行的平均值
    mmcar_avg = np.column_stack([np.mean(matrix, axis=1) for matrix in [US_D, AS_D, US_C, AS_C, AI, UR, AR, UV, AV]])

    return mmcar_avg



def run_simulation(i, epidemic_sim):
    epidemic = copy.deepcopy(epidemic_sim)
    print("第 %s 次MC模拟" % (i + 1))
    epidemic.init_awareness(init_u=0.99)
    epidemic.init_strategy(init_c=0.1)
    epidemic.init_state()
    mc = []  # 保存未感染的初始态
    epidemic.init_infect(init_i=0.02)  # 感染初始化
    mc.append(epidemic.count_all())  # 保存随机感染后各状态密度

    for t in range(0, 29):
        epidemic.MC_Simulation()
        mc.append(epidemic.count_all())

    return np.array(mc)


def main():

    # 创建项目文件夹
    current_file_path = Path(__file__).parent
    relative_path = Path("project_1")
    absolute_path = current_file_path / relative_path
    os.makedirs(absolute_path, exist_ok=True)

    seed = 3
    lower_net = nx.barabasi_albert_graph(500, 5, seed = int(seed * np.random.rand() * 100 ))
    upper_net = Game.add_random_edges(lower_net, 200)

    Epidemic_sim = Epidemic.Epidemic(lower_net,
                                     upper_net,
                                     alpha=0.6, delta=0.4, beta=0.83333, eff=0.8, omega=0.25, eta=0.6,
                                     gamma=0.3333)
    # 有修正项MMCA理论计算
    mmca_avg = calculate_mmca_avg(Epidemic_sim)
    df_mmca = pd.DataFrame(mmca_avg)
    df_mmca.to_csv(absolute_path/f"avg_mmca.csv", index=False, header=False)

    # 无修正项MMCA理论计算
    mmcar_avg = calculate_mmcar_avg(Epidemic_sim)
    df_mmcar = pd.DataFrame(mmcar_avg)
    df_mmcar.to_csv(absolute_path / f"avg_mmcar.csv", index=False, header=False)

    # MC模拟
    num_sims = 32
    mc_sums = None
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_simulation, i, Epidemic_sim) for i in range(num_sims)]

        for future in as_completed(futures):
            mc_result = future.result()
            if mc_sums is None:
                mc_sums = mc_result
            else:
                mc_sums += mc_result
    mc_avg = mc_sums / num_sims
    df_mc_avg = pd.DataFrame(mc_avg)
    df_mc_avg.to_csv(absolute_path/f"avg_mc.csv", index=False, header=False)


if __name__ == "__main__":
    main()