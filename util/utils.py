import numpy as np
import os

def save_statistic(agent_info, env_info, traffic, utilization, num_sample, reward):
    headers = 't_lower_q,t_higher_q,t_median,t_max,t_min,t_mean,t_std,u_lower_q,u_higher_q,u_median,u_max,u_min,u_mean,u_std,congestion,throughput,num_sample,reward'
    t_lower_q = np.quantile(traffic, 0.25, interpolation='lower')  # 下四分位数
    t_higher_q = np.quantile(traffic, 0.75, interpolation='higher')  # 上四分位数
    t_median = np.median(traffic)
    t_max = np.max(traffic)
    t_min = np.min(traffic)
    t_mean = np.mean(traffic)
    t_std = np.std(traffic)

    u_lower_q = np.quantile(utilization, 0.25, interpolation='lower')  # 下四分位数
    u_higher_q = np.quantile(utilization, 0.75, interpolation='higher')  # 上四分位数
    u_median = np.median(utilization)
    u_max = np.max(utilization)
    u_min = np.min(utilization)
    u_mean = np.mean(utilization)
    u_std = np.std(utilization)

    congestion = sum(i >= 1 for i in utilization) / len(utilization)

    throughput = np.sum(traffic)

    result_dir = 'result/work'
    path = os.path.join(result_dir, agent_info + env_info)

    statistic_data = str(t_lower_q) + ',' + str(t_higher_q) + ',' + str(t_median) + ',' + str(t_max) + ',' + str(
        t_min) + ',' + str(t_mean) + ',' + str(t_std) + ',' + str(u_lower_q) + ',' + str(u_higher_q) + ',' + str(
        u_median) + ',' + str(u_max) + ',' + str(u_min) + ',' + str(u_mean) + ',' + str(u_std) + ',' + str(
        congestion) + ',' + str(throughput) + ',' + str(num_sample) + ',' + str(reward)

    if os.path.exists(path + '-statistic.csv') == False:
        with open(path + '-statistic.csv', 'w') as f:
            f.write(headers + '\n')
            f.write(statistic_data + '\n')
            f.close()
    else:
        with open(path + '-statistic.csv', 'a') as f:
            f.write(statistic_data + '\n')
            f.close()


def save(agent_info, iteration, reward, throughput, min_utilization, mean_utilization):
    # throughput:评估次数*1-计算N次评估iteration的平均吞吐量，最大吞吐量，最小吞吐量
    # min_utilization:评估次数*1-每个iteration内最小链路率取平均
    # mean_utilization:评估次数*1-所有链路的平均路利用率取最大、平均、最小
    result_dir = 'result/work'
    path = os.path.join(result_dir, agent_info)

    min_put = np.min(throughput)
    mean_put = np.mean(throughput)
    max_put = np.max(throughput)
    min_ut = np.mean(min_utilization)
    min_m_ut = np.min(mean_utilization)
    mean_m_ut = np.mean(mean_utilization)
    max_m_ut = np.max(mean_utilization)

    statistic_data = str(iteration) + ',' + str(reward)+str(min_put)+ str(mean_put)+ str(max_put)+str(
        min_ut) +str(min_m_ut)+ str(mean_m_ut)+ str(max_m_ut)

    if not os.path.exists(path + '-statistic.csv'):
        with open(path + '-statistic.csv', 'w') as f:
            # f.write(headers + '\n')
            f.write(statistic_data + '\n')
            f.close()
    else:
        with open(path + '-statistic.csv', 'a') as f:
            f.write(statistic_data + '\n')
            f.close()


def save_reward(agent_info, iteration, reward):

    result_dir = 'result'
    path = os.path.join(result_dir, agent_info)

    statistic_data = str(iteration) + ',' + str(reward)

    if not os.path.exists(path + '-statistic.csv'):
        with open(path + '-statistic.csv', 'w') as f:
            f.write(statistic_data + '\n')
            f.close()
    else:
        with open(path + '-statistic.csv', 'a') as f:
            f.write(statistic_data + '\n')
            f.close()

def save_throughput(agent_info, iteration, reward, throughput, min_utilization, mean_utilization):
    # throughput:评估次数*1-计算N次评估iteration的平均吞吐量，最大吞吐量，最小吞吐量
    # min_utilization:评估次数*1-每个iteration内最小链路率取平均
    # mean_utilization:评估次数*1-所有链路的平均路利用率取最大、平均、最小
    result_dir = 'result/res'
    path = os.path.join(result_dir, agent_info)

    min_put = np.min(throughput)
    mean_put = np.mean(throughput)
    max_put = np.max(throughput)
    min_ut = np.mean(min_utilization)
    min_m_ut = np.min(mean_utilization)
    mean_m_ut = np.mean(mean_utilization)
    max_m_ut = np.max(mean_utilization)

    statistic_data = str(iteration) + ',' + str(reward)+ ',' +str(min_put)+ ',' + str(mean_put)+ ',' + str(max_put)+ \
                     ',' +str(min_ut) + ',' +str(min_m_ut)+ ',' + str(mean_m_ut)+ ',' + str(max_m_ut)

    if not os.path.exists(path + '-statistic.csv'):
        with open(path + '-statistic.csv', 'w') as f:
            # f.write(headers + '\n')
            f.write(statistic_data + '\n')
            f.close()
    else:
        with open(path + '-statistic.csv', 'a') as f:
            f.write(statistic_data + '\n')
            f.close()
