import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import time
import platform
import sys
import ETPRooms

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

print('系统:', platform.system())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU提升运算速度
print(device)

room_number = 2  # 房间的数量

ki_1 = 6425
kl_1 = 1380
ko_1 = 1623
Ca_1 = 18.74 * 10 ** 6
Cm_1 = 844 * 10 ** 6  # for hotel

ki_2 = 725
kl_2 = 97
ko_2 = 312
Ca_2 = 1.44 * 10 ** 6
Cm_2 = 466 * 10 ** 6  # for retail

ki_list = [ki_1, ki_2]
kl_list = [kl_1, kl_2]
ko_list = [ko_1, ko_2]
ca_list = [Ca_1, Ca_2]
cm_list = [Cm_1, Cm_2]
# 此处将房间的参数整理为列表，方便网络类进行处理与初始化。若拓展至更多房间，该参数需要进行修改

dt = 1 / 60 / 15  # 时间微元为4秒
repeat_len = int(24 / dt)  # 重复的步数
COP = 3  # 电功率热功率转化系数

# 文件读取
t0 = time.perf_counter()
Shanghai_data = pd.read_excel(io='./CHN_Shanghai.Shanghai.583670_IWEC.xlsx')  # 温度数据
To_list = []
for i in Shanghai_data.index.values:
    row_data = Shanghai_data.loc[i, ['Dry Bulb Temperature']].to_dict()
    row_data_0 = list(row_data.values())
    To_list.append(row_data_0[0])
t1 = time.perf_counter()
print('文件读取完毕！用时 {:.3f} 秒'.format(t1 - t0))

Q_1 = []
Q_2 = []

To_record = []
start_day = 0

Ti_ini_list = [To_list[start_day * 24], To_list[start_day * 24]]
Ti_1_list = [To_list[start_day * 24] + 2]
Ti_2_list = [To_list[start_day * 24] + 2]

Q_reg_list = []

time_length = 60 * 4 * 15  # 总时长，设为12h，每4秒为1个时段
Ti_set_1 = 25  # 房间1设定温度
Ti_set_2 = 24  # 房间2设定温度
Ti_ref_list = [Ti_set_1, Ti_set_2]

stable_power_list = []
To_sum = 0
for t0 in range(24):
    To_sum += To_list[start_day * 24 + t0]
To_avg = To_sum / 24
for r_i in range(room_number):
    stable_power = ((ki_list[r_i] * kl_list[r_i] + ki_list[r_i] * ko_list[r_i] + kl_list[r_i] * ko_list[r_i])
                    * (Ti_ref_list[r_i] - To_avg) / (kl_list[r_i] + ko_list[r_i])) / COP
    stable_power_list.append(abs(stable_power) // 1000 * 1000)
print('整定后的基准电功率(W)' + str(stable_power_list))

# 强化学习参数
GAMMA = 0.98
ACTOR_LR = 0.00001
CRITIC_LR = 0.00001
TAU = 0.0001
CAPACITY = 1000000
BATCH_SIZE = 128
EPOCH = 200
MAX_STEP = 60 * 8 * 15
TRAIN_MODE = True
SAVE_PTH = "MATD3_Agents/room"
INI_VARIANCE = 0.2
DELAY_TIME = 2

MyNet = ETPRooms.AreaNet(room_number, ki_list, kl_list, ko_list, ca_list, cm_list, Ti_ini_list, stable_power_list,
                         GAMMA, ACTOR_LR, CRITIC_LR, TAU, CAPACITY, SAVE_PTH, INI_VARIANCE, DELAY_TIME)

"""
室外温度线性插值，让仿真效果更接近实际情况
"""
To_need_list = [To_need for To_need in To_list[start_day:int(max(repeat_len, MAX_STEP) * dt) + 1 + start_day]]
To_x = [to_x for to_x in range(int(max(repeat_len, MAX_STEP) * dt) + 1)]
To_time = np.linspace(0, int(max(repeat_len, MAX_STEP) * dt) + 1, num=max(MAX_STEP, repeat_len))
To_new_list = np.interp(To_time, To_x, To_need_list)

if TRAIN_MODE:  # 训练模式下，直接开启训练并生成训练模型
    reward = []
    time_span = []
    total_step = 0
    max_episode_reward = 0
    max_reward_episode = -1000
    best_model_file_pth = "MATD3_Agents/best_room"
    MyNet.Cen_Agent.save_path = best_model_file_pth
    for episode in range(EPOCH):  # 训练过程
        Ti_train_set = Ti_ref_list
        episode_reward = 0
        MyNet.area_reset()
        time_start = time.perf_counter()
        if total_step % (60 * 15 * 16) == 0 and total_step >= MAX_STEP * 10:
            if random.random() > 0.5:
                for ri in range(room_number):
                    MyNet.room_list[ri].ti += random.randint(-5, 25)
                    MyNet.room_list[ri].tm = MyNet.room_list[ri].ti + random.randint(-2, 2)
        for step in range(MAX_STEP):
            total_step += 1
            if step == MAX_STEP - 1:
                MyNet.done = True
            Q_room, r = MyNet.env_step(Ti_train_set, To_new_list[step])
            episode_reward += r
            if len(MyNet.Replay_Buffer.buffer) > BATCH_SIZE * 50:
                sample = MyNet.Replay_Buffer.sample(BATCH_SIZE)
                MyNet.Cen_Agent.learn(sample)

            print('\r', end='')
            print("learn:[", end="")
            print("#" * (((step + 1) * 10) // MAX_STEP), end="")
            print(" " * (10 - ((step + 1) * 10) // MAX_STEP), end="]")
            print("{}/{}, epoch:{}".format(step + 1, MAX_STEP, episode + 1), end="")
            sys.stdout.flush()
            if total_step % (10 * MAX_STEP) == 10 * MAX_STEP - 1 and total_step >= 20 * MAX_STEP:
                if total_step <= 100 * MAX_STEP:
                    MyNet.Cen_Agent.variance /= 2  # 减小噪声方差
                elif total_step >= 160 * MAX_STEP:
                    MyNet.Cen_Agent.variance /= 5
                else:
                    MyNet.Cen_Agent.variance *= 4  # 退火操作，减小局部最优可能性
                    for room_i in range(room_number):
                        for p1 in MyNet.Cen_Agent.agents[room_i].critic_1_optim.param_groups:  # 动态调整学习率，提升学习效果
                            p1['lr'] *= 25
                        for p2 in MyNet.Cen_Agent.agents[room_i].critic_2_optim.param_groups:
                            p2['lr'] *= 25
                        for p3 in MyNet.Cen_Agent.agents[room_i].actor_optim.param_groups:
                            p3['lr'] *= 25

                for room_i in range(room_number):
                    for p1 in MyNet.Cen_Agent.agents[room_i].critic_1_optim.param_groups:  # 动态调整学习率，提升学习效果
                        p1['lr'] *= 0.2
                    for p2 in MyNet.Cen_Agent.agents[room_i].critic_2_optim.param_groups:
                        p2['lr'] *= 0.2
                    for p3 in MyNet.Cen_Agent.agents[room_i].actor_optim.param_groups:
                        p3['lr'] *= 0.2

        reward.append(episode_reward)
        time_end = time.perf_counter()
        time_consume = time_end - time_start
        time_span.append(time_consume)
        # print("============episode:{}/{}============".format(episode + 1, EPOCH))
        print(', time use:{:.1f}s, reward_sum:{:.3f}'.format(time_consume, float(episode_reward)))
        if episode_reward >= max_episode_reward:
            max_episode_reward = episode_reward
            max_reward_episode = episode + 1
            MyNet.Cen_Agent.save_td3()  # 保存总体性能最优的模型

    MyNet.Cen_Agent.save_path = SAVE_PTH
    MyNet.Cen_Agent.save_td3()  #保存最终的模型
    t2 = time.perf_counter()
    train_hour = int((t2 - t1) // 3600)
    train_minute = int((t2 - t1) // 60) - train_hour * 60
    train_second = int((t2 - t1) % 60)
    print(
        f'模型训练完毕！用时{train_hour:d}时{train_minute:d}分{train_second:d}秒，最优奖励为{max_episode_reward:3f}，出现在第{max_reward_episode:d}次迭代')

    # 显示训练曲线
    plt.subplot(2, 1, 1)
    plt.plot(reward, linewidth=0.6, color='green')
    plt.subplot(2, 1, 2)
    plt.plot(time_span, linewidth=0.6, color='red')  # 时间性能分析
    plt.show()
    t2 = time.perf_counter()  # 显示训练曲线时程序会暂停，因此需要刷新一下时间

else:  # 非训练模式下，需从训练好的文件中读取数据
    MyNet.Cen_Agent.load_td3()
    t2 = time.perf_counter()
    print('模型加载完毕！用时 {:.3f} 秒'.format(t2 - t1))

MyNet.area_reset()
MyNet.Cen_Agent.variance = 1e-4

for t in range(time_length):
    Q_list, r, = MyNet.env_step(Ti_ref_list, To_new_list[t])
    # 根据TD3算功率
    Q_1.append(Q_list[0])
    Q_2.append(Q_list[1])
    Ti_1_list.append(MyNet.room_list[0].ti)
    Ti_2_list.append(MyNet.room_list[1].ti)
    To_record.append(To_new_list[t])

t3 = time.perf_counter()
print('数据处理完毕！用时 {:.3f} 秒'.format(t3 - t2))

# 作图
Ti_set_1_line = Ti_set_1 * np.ones(time_length)
Ti_set_2_line = Ti_set_2 * np.ones(time_length)

fig = plt.figure()
ax1 = fig.add_subplot(5, 1, 1)
ax2 = fig.add_subplot(5, 1, 3)
ax3 = fig.add_subplot(5, 1, 5)

# 第一张图绘制房间1的室温与墙壁温度
ax1.plot(Ti_1_list, linewidth=0.6, label='Ti1')
ax1.plot(To_record, linewidth=0.6, label='To')
ax1.plot(Ti_set_1_line, linewidth=0.3, linestyle='--')
ax1.legend()
ax1.set_title('Room1')
ax1.set_xlabel('Time/4sec')
ax1.set_ylabel('Temp/℃')

# 第二张图绘制房间2的室温与墙壁温度
ax2.plot(Ti_2_list, linewidth=0.6, label='Ti2')
ax2.plot(To_record, linewidth=0.6, label='To')
ax2.plot(Ti_set_2_line, linewidth=0.3, linestyle='--')
ax2.legend()
ax2.set_title('Room2')
ax2.set_xlabel('Time/4sec')
ax2.set_ylabel('Temp/℃')

# 第三张图绘制功率曲线，以电功率为基准
P_list = []
Qe1 = []
Qe2 = []
for i in range(time_length):
    P_list.append(abs(Q_1[i]) / COP + abs(Q_2[i]) / COP)
    Qe1.append(abs(Q_1[i]) / COP)
    Qe2.append(abs(Q_2[i]) / COP)
ax3.plot(Qe1, linewidth=0.8, label='Q1')
ax3.plot(Qe2, linewidth=0.8, label='Q2')
ax3.plot(P_list, linewidth=0.8, label='Q_total')
ax3.legend()
ax3.set_title('Electricity Power')
ax3.set_xlabel('Time/4sec')
ax3.set_ylabel('Power/W')

plt.show()
