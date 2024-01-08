import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import platform
print('系统:', platform.system())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #使用GPU提升运算速度
#有个很关键的问题：实际场景用GPU是否合理？一般来说CPU更适合用在各种控制器上面
print(device)

room_number = 2  # 房间的数量

ki_1 = 6425
kl_1 = 1380
ko_1 = 1623
Ca_1 = 18.74*10**6
Cm_1 = 844*10**6  # for hotel

ki_2 = 725
kl_2 = 97
ko_2 = 312
Ca_2 = 1.44*10**6
Cm_2 = 466*10**6  # for retail

ki_list = [ki_1, ki_2]
kl_list = [kl_1, kl_2]
ko_list = [ko_1, ko_2]
ca_list = [Ca_1, Ca_2]
cm_list = [Cm_1, Cm_2]
# 此处将房间的参数整理为列表，方便网络类进行处理与初始化。若拓展至更多房间，该参数需要进行修改

dt = 1 / 120   #时间微元为半分钟
repeat_len = int(24/dt)    #重复的步数
COP = 3  #电功率热功率转化系数


"""
采用算法：TD3（Twin Delayed Deep Deterministic policy gradient algorithm，双延迟深度确定性策略梯度）
TD3算法建立在DDPG算法的基础上，而DDPG算法是建立在DQN基础上的，使用Actor输出动作，使用Critic输出q值，TD3与DDPG可以用来处理连续性问题
本项目预期功率属于连续值，同时状态量中的温度与regD功率也属于连续量，因此采用可以处理连续问题的强化学习算法
DDPG在单房间模型的温度控制上有一定的效果，本代码使用表现更好的TD3算法，实现温度控制与regD信号跟踪的双重效果
此版本代码是纯TD3方法的升级版，采用连续多个时刻的状态作为神经网络的状态输入，从而更好地体现当前是升温状态还是降温状态
"""


#Actor网络，用于输出动作
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 网络初始化：输入层、隐藏层、输出层
        # 输入层代表输入的状态维度，输出层代表动作的维度，隐藏层则用于建立连接
        super(Actor, self).__init__()
        # 网络连接层初始化，这里采用3个线性层
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear2.weight.data.normal_(0, 0.02)
        self.linear3 = nn.Linear(int(hidden_size/2), output_size)
        self.linear3.weight.data.normal_(0, 0.02)

    def forward(self, s):
        # 前向函数设立，将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止
        # fc.elu()函数：elu(x) = max(0, x)
        # torch.tanh()函数：(e**x-e**(-x))/(e**x+e**(-x))
        x = fc.elu(self.linear1(s))
        x = fc.elu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


# Critic网络，用于输出q值，从而影响动作的选择
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 网络初始化：输入层、隐藏层、输出层
        # 输入层代表输入的状态维度，输出层代表动作的维度，隐藏层则用于建立连接
        super().__init__()
        # 网络连接层初始化，这里采用3个线性层
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear2.weight.data.normal_(0, 0.01)
        self.linear3 = nn.Linear(int(hidden_size/2), output_size)
        self.linear3.weight.data.normal_(0, 0.01)

    def forward(self, s, a):
        # 前向函数设立，将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止
        # fc.relu()函数：relu(x) = max(0, x)
        x = torch.cat([s, a], 1)
        x = fc.relu(self.linear1(x))
        x = fc.sigmoid(self.linear2(x))
        x = self.linear3(x)

        return x


# Agent，用于执行强化学习的任务
class Agent:
    def __init__(self, room_num, gamma, actor_lr, critic_lr, tau, capacity, batch_size, save_path,
                 norm_variance=0.1, noise_scale=0.2, delay_time=2):
        self.gamma = gamma  # 每次迭代奖励值的折扣系数
        self.actor_lr = actor_lr  # Actor网络的学习率
        self.critic_lr = critic_lr  # Critic网络的学习率
        self.tau = tau  # 网络软更新参数
        self.capacity = capacity  # 内存池容量
        self.batch_size = batch_size  # 每次更新的容量
        self.save_path = save_path  #文件保存路径
        self.variance = norm_variance  # 高斯噪声方差
        self.update_time = 0  # 更新次数，用于Actor的延迟更新策略
        self.delay_time = delay_time  # 延迟更新时间，用于Actor的延迟更新策略
        self.room_num = room_num
        self.noise_scale = noise_scale

        self.s_dim = room_num * 3 + 1  # 每个房间每一步的状态维度有3个：Ti, Tm和Ti_ref，整个园区额外有1个状态: To
        self.a_dim = room_num  # 每个房间的动作维度仅1个，即热功率

        # 神经网络的初始化，需要建立6个网络：Actor, Critic_1, Critic_2, Actor的目标网络， Critic_1的目标网络, Critic_2的目标网络
        # Critic采用双重网络的目的是选择q值预测值更小的输出，减小Q值预测值被高估的问题
        # 采用目标网络可以避免Q值的更新发生震荡
        self.actor = Actor(self.s_dim*2, 32, self.a_dim)
        self.actor.to(device)
        self.actor_target = Actor(self.s_dim*2, 32, self.a_dim)
        self.actor_target.to(device)
        self.critic_1 = Critic(self.s_dim*2 + self.a_dim, 128, self.a_dim)
        self.critic_1.to(device)
        self.critic_2 = Critic(self.s_dim*2 + self.a_dim, 144, self.a_dim)
        self.critic_2.to(device)
        self.critic_1_target = Critic(self.s_dim*2 + self.a_dim, 128, self.a_dim)
        self.critic_1_target.to(device)
        self.critic_2_target = Critic(self.s_dim*2 + self.a_dim, 144, self.a_dim)
        self.critic_2_target.to(device)
        # 神经网络优化器以及经验回放池的建立
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optim = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optim = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        first_info = np.zeros(self.s_dim+self.a_dim+2)
        self.buffer = np.array(first_info)
        # 初始化状态下将神经网络本体与目标网络进行同步
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def act(self, state):
        # 根据智能体目前的状态选择最优的动作
        state = torch.tensor(state, dtype=torch.float).to(device).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().cpu().numpy()  # 借助Actor网络选择最优动作
        return action

    def put(self, *transition):
        # 将新的状态、动作与奖励存储进经验回放池中
        # 如果经验回放池已满，则将最早的经验记录清除
        if len(self.buffer) == self.capacity:
            np.delete(self.buffer, 0)  # 似乎是耗时很长的关键原因，长数组的弹出栈和堆栈很耗时
        self.buffer = np.vstack((self.buffer, transition))  # 将新的机器学习内容放入回放池

    def learn(self):
        if len(self.buffer)-2 < self.batch_size:  # 经验池中内容若小于批量更新数，则先不进行学习
            return

        #从经验回放池中随机采样，通过选取参数进行，便于直接从经验池中获取下一时刻的数据
        index = np.random.choice(len(self.buffer)-2, size=self.batch_size, replace=True) + 1
        sample = self.buffer[index, :]
        sample_next = self.buffer[index+1, :]
        sample_last = self.buffer[index-1, :]

        #采样的内容包含此时刻的动作、奖励与状态以及下一时刻的状态，额外增加是否截断
        #由于buffer里面单条存储内容不包含下一时刻的s，因此r1与s1的表示方式也要改变，否则无法达到理想效果！
        s1 = sample[:, :self.s_dim]
        a_now = sample[:, self.s_dim:self.s_dim+self.a_dim]
        r_this = sample[:, self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        d_this = sample[:, self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+2]
        s2 = sample_next[:, :self.s_dim]
        s_last = sample_last[:, :self.s_dim]
        d_last = sample_last[:, self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+2]
        # 将状态与动作转化为pytorch可以处理的张量形式
        s1 = np.array(s1)
        a_now = np.array(a_now)
        r_this = np.array(r_this)
        d_this = np.array(d_this)
        s_next = np.array(s2)
        s_last = np.array(s_last)
        d_last = np.array(d_last)
        if d_last[0] == 1:
            s_now = np.append(s_last, s1, axis=1)
        else:
            s_now = np.append(s1, s1, axis=1)
        s_next = np.append(s1, s2, axis=1)
        s_now = torch.tensor(s_now, dtype=torch.float).to(device)
        a_now = torch.tensor(a_now, dtype=torch.float).to(device)
        r_this = torch.tensor(r_this, dtype=torch.float).view(self.batch_size, -1).to(device)
        d_this = torch.tensor(d_this, dtype=torch.float).view(self.batch_size, -1).to(device)
        s_next = torch.tensor(s_next, dtype=torch.float).to(device)

        def critic_learn():
            # Critic网络的学习与更新
            noise = torch.tensor(np.clip(np.random.normal(0, self.variance, size=self.room_num),
                                         -self.noise_scale/2, self.noise_scale/2), dtype=torch.float).to(device)
            a1 = torch.clamp(self.actor_target(s_next).detach() + noise, -1, 1)
            # 选出下一时刻最优化的动作，并且加上噪声
            # 噪声我们依然采用正态分布，但是需要进行截断，偶然出现的大偏差随机值也是导致过估计的重要原因之一

            q1 = self.critic_1_target(s_next, a1).detach()  # 根据目标网络定义y值，作为更新网络的依据
            q2 = self.critic_2_target(s_next, a1).detach()
            q_true = torch.min(q1, q2)  # 选取最小的q值，减小过估计的问题
            y_true = r_this + self.gamma * q_true * d_this

            y_1_predict = self.critic_1(s_now, a_now)  # 从Critic网络中读取q值作为y预测值
            y_2_predict = self.critic_2(s_now, a_now)

            loss_fn = nn.MSELoss()  # 定义损失函数MSELoss： 输入x（模型预测输出）和目标y之间的均方误差标准
            #  通过使损失函数最小化寻找最优值以更新Critic函数
            loss_1 = loss_fn(y_1_predict, y_true)
            loss_2 = loss_fn(y_2_predict, y_true)
            loss = loss_1 + loss_2
            self.critic_1_optim.zero_grad()
            self.critic_2_optim.zero_grad()
            loss.backward()
            self.critic_1_optim.step()
            self.critic_2_optim.step()

        def actor_learn():
            # Actor网络的学习与更新
            loss = -torch.mean(self.critic_1(s_now, self.actor(s_now)))
            # 利用采样后的策略梯度的均值计算损失函数，并更新Actor网络
            # 对于actor来说，其实并不在乎Q值是否会被高估，他的任务只是不断做梯度上升，寻找这条最大的Q值。
            # 随着更新的进行，Q1和Q2两个网络将会变得越来越像。所以用Q1还是Q2，还是两者都用，对于actor的问题不大。
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            # 神经网络软更新，使每次迭代都更新目标网络时，算法也能保持一定的稳定性
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        # 进行网络的学习与更新
        critic_learn()
        self.update_time += 1
        if self.update_time % self.delay_time == 0:  # Actor网络延迟更新策略，减少局部最优解的可能性
            actor_learn()
            soft_update(self.critic_1_target, self.critic_1, self.tau)  # 更新Actor网络以后再进行目标网络的软更新
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

    def load_td3(self):
        # 调用现成的模型文件，方便直接应用
        try:
            checkpoint = torch.load(self.save_path)
            self.actor.load_state_dict(checkpoint['actor_net_dict'])
            self.critic_1.load_state_dict(checkpoint['critic_1_net_dict'])
            self.critic_2.load_state_dict(checkpoint['critic_2_net_dict'])
            self.actor_optim.load_state_dict(checkpoint['actor_net_dict'])
            self.critic_1_optim.load_state_dict(checkpoint['critic_1_net_dict'])
            self.critic_2_optim.load_state_dict(checkpoint['critic_2_net_dict'])
        except Exception as e:
            print(e)

    def save_td3(self):
        # 保存训练模型，在训练完以后便于直接调用训练文件
        torch.save(
            {
                'actor_net_dict': self.actor.state_dict(),
                'critic_1_net_dict': self.critic_1.state_dict(),
                'critic_2_net_dict': self.critic_2.state_dict(),
                'actor_optim_dict': self.actor_optim.state_dict(),
                'critic_1_optim_dict': self.critic_1_optim.state_dict(),
                'critic_2_optim_dict': self.critic_2_optim.state_dict(),
            }, self.save_path)


class AreaNet:
    """
    该类为园区网络模型的模型构建，用于处理各个房间之间的交互数据
    参数：
    room_num：房间的数量
    ki_lst, kl_lst, ko_lst, ca_lst, cm_lst：房间参数列表
    ti_ini_lst, tm_ini_lst：房间初始温度参数列表
    center_power：中心功率，即RegD信号基准功率
    gamma, actor_lr, critic_lr, tau, capacity, batch_size, norm_variance, noise_scale, delay：TD3强化学习参数
    save_pth：强化学习模型文件保存路径
    """

    def __init__(self, room_num, ki_lst, kl_lst, ko_lst, ca_lst, cm_lst, ti_ini_lst, tm_ini_lst, center_power,
                 gamma, actor_lr, critic_lr, tau, capacity, batch_size, save_pth,
                 norm_variance=0.1, noise_scale=0.2, delay=2):
        if np.all([len(ki_lst), len(kl_lst), len(ko_lst), len(ca_lst), len(cm_lst),
                   len(ti_ini_lst), len(tm_ini_lst)] == room_num):
            print([len(ki_lst), len(kl_lst), len(ko_lst), len(ca_lst), len(cm_lst),
                   len(ti_ini_lst), len(tm_ini_lst)])
            print(room_num)
            raise ValueError(
                "Room number is not matched with room parameter lists!"
            )
        # 先判断房间的数量是否与参数匹配，若不匹配则报错，避免因为参数不匹配导致后续的运算错误

        self.room_n = room_num
        self.room_list = [
            self.ETPRoom(ki_lst[room_index], kl_lst[room_index], ko_lst[room_index], ca_lst[room_index],
                         cm_lst[room_index], ti_ini_lst[room_index], tm_ini_lst[room_index])
            for room_index in range(room_num)]  # ETP模型为基础的房间初始化
        self.ti_list = ti_ini_lst
        self.tm_list = tm_ini_lst
        self.q_list = np.zeros(room_num)  # 园区各个房间的状态列表
        self.Agent = Agent(room_num, gamma, actor_lr, critic_lr, tau, capacity, batch_size,
                           save_pth, norm_variance, noise_scale, delay)  # 中央控制器的初始化
        self.power_limit = center_power * 2  # 功率上限
        self.done = False  # 一个训练周期是否结束的标识

    class ETPRoom:
        """
        该类用于单个房间的模型构建，便于处理多个房间的问题
        参数：
        ki, kl, ko, ca, cm ： 房间的ETP模型参数
        ti_ini, tm_ini ： 房间的初始温度
        """

        def __init__(self, ki, kl, ko, ca, cm, ti_ini, tm_ini):
            # 参数的初始化
            self.ki = ki
            self.kl = kl
            self.ko = ko
            self.ca = ca
            self.cm = cm
            self.ti = ti_ini
            self.tm = tm_ini
            self.ti_ini = ti_ini
            self.tm_ini = tm_ini

        def reset(self):
            self.ti = self.ti_ini
            self.tm = self.tm_ini

        def cal_temp(self, to_last, q_last):
            # 房间温度的预测
            self.ti = self.ti + dt * 3600 * 15 * (
                    self.ki * (self.tm - self.ti) + self.kl * (to_last - self.ti) + q_last) / self.ca
            self.tm = self.tm + dt * 3600 * 15 * (
                    self.ki * (self.ti - self.tm) + self.ko * (to_last - self.tm)) / self.cm
            return self.ti, self.tm

    def temp_update(self, to_last):
        # 根据ETP模型算出每个房间的室温与墙壁温度
        for room_i in range(self.room_n):
            self.ti_list[room_i], self.tm_list[room_i] = self.room_list[room_i].cal_temp(to_last, self.q_list[room_i])
        return self.ti_list, self.tm_list

    def area_reset(self):
        # 重置园区房间的环境
        for room in self.room_list:
            room.reset()
        self.done = False

    def power_allocation(self, ti_ref_list, t_o, s_last):
        # 中央控制器对园区内房间进行功率分配，并更新温度，同时负责进行强化学习数据的存储
        state_now = []
        for room_i in range(self.room_n):
            state_now.append(self.room_list[room_i].ti/50)
            state_now.append(self.room_list[room_i].tm/50)
            state_now.append(ti_ref_list[room_i]/50)
        state_now.append(t_o/50)    # 状态量，包含每个房间的各类温度，归一化处理
        s0 = state_now
        state_now = np.append(state_now, s_last)
        action_now = self.Agent.act(state_now)   # 动作量，包含每个房间的空调功率，归一化处理
        self.q_list = action_now.astype(np.float32) * self.power_limit
        temp_next, _ = self.temp_update(t_o)    # 房间温度与状态的更新
        reward_now = 0
        all_fit_flag = 1
        for room_i in range(self.room_n):
            if abs(self.room_list[room_i].ti - ti_ref_list[room_i]) > 1:
                reward_now -= abs(self.room_list[room_i].ti - ti_ref_list[room_i])**2 / 625 + 0.1
                all_fit_flag = 0
            else:
                reward_now += 0.02
        if all_fit_flag == 1:
            reward_now += 0.1 * self.room_n
        # 奖赏值定义，取决于温度
        if not self.done:
            d1 = 1
        else:
            d1 = 0   # 是否为一个训练周期内末尾的数据，末尾数据被裁切，避免产生错误的学习数据
        trans0 = np.append(s0, action_now)
        trans1 = np.append(trans0, reward_now)
        trans = np.append(trans1, d1)
        self.Agent.put(trans)    # 将状态量、动作量与奖赏值合并，放入智能体的经验回放池中便于学习

        return self.q_list, reward_now, s0   # 返回值是各个房间的功率列表与智能体得到的奖赏值


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

Ti_ini_list = [To_list[start_day*24], To_list[start_day*24]]
Tm_ini_list = [To_list[start_day*24], To_list[start_day*24]]
Ti_1_list = [To_list[start_day*24]]  #从5月4日开始进行仿真；数据的年份并不匹配，但是仍然具有合理性
Ti_2_list = [To_list[start_day*24]]
Tm_1_list = [To_list[start_day*24]]
Tm_2_list = [To_list[start_day*24]]

time_length = 60 * 48  # 总时长，设为24h，每1min为1个时段
Ti_set_1 = 25  #房间1设定温度
Ti_set_2 = 24  #房间2设定温度
Ti_ref_list = [Ti_set_1, Ti_set_2]

stable_power_sum = 0
To_sum = 0
for t0 in range(24):
    To_sum += To_list[start_day * 24 + t0]
To_avg = To_sum/24
stable_power_list = []
for r_i in range(room_number):
    stable_power = ((ki_list[r_i] * kl_list[r_i] + ki_list[r_i] * ko_list[r_i] + kl_list[r_i] * ko_list[r_i])
                    * (Ti_ref_list[r_i] - To_avg) / (kl_list[r_i] + ko_list[r_i])) / COP
    stable_power_sum += abs(stable_power)
    stable_power_list.append(abs(stable_power))
base_power = stable_power_sum // 1000 * 1000
print('整定后的基准电功率：{:.1f}W'.format(base_power))

# 强化学习参数
GAMMA = 0.99
ACTOR_LR = 0.00004
CRITIC_LR = 0.0002
TAU = 0.005
CAPACITY = 48000
BATCH_SIZE = 32
EPOCH = 120
MAX_STEP = 60 * 12 * 2
TRAIN_MODE = True
SAVE_PTH = "TD3_no_reg.pth"
INI_VARIANCE = 1
DELAY_TIME = 5

MyNet = AreaNet(room_number, ki_list, kl_list, ko_list, ca_list, cm_list, Ti_ini_list, Tm_ini_list, base_power,
                GAMMA, ACTOR_LR, CRITIC_LR, TAU, CAPACITY, BATCH_SIZE, SAVE_PTH, INI_VARIANCE, DELAY_TIME)

if TRAIN_MODE:  # 训练模式下，直接开启训练并生成训练模型
    reward = []
    time_span = []
    total_step = 0
    for episode in range(EPOCH):  # 训练过程
        # Ti_train_set = random.randint(180, 300) / 10  # 训练时随机设定目标温度，从而使模型有更好的普适性
        Ti_train_set = Ti_ref_list  # 老老实实先练好当前的再说
        episode_reward = 0
        MyNet.area_reset()
        time_start = time.perf_counter()
        state_last = []
        for i in range(room_number):
            state_last.append(Ti_ini_list[i]/50)
            state_last.append(Tm_ini_list[i]/50)
            state_last.append(Ti_ref_list[i]/50)
        state_last.append(To_list[0])
        for step in range(MAX_STEP):
            '''
            if step % (60 * 15) == 0 and step != 0:
                # 训练时每1h更新一次设定温度，保证训练效果的同时减少需要的迭代次数
                Ti_train_set = random.randint(180, 300) / 10
            '''
            total_step += 1
            if step == MAX_STEP - 1:
                MyNet.done = True
            Q_room, r, state_last = MyNet.power_allocation(Ti_train_set, To_list[int((step - 1) * dt)], state_last)
            episode_reward += r
            MyNet.Agent.learn()
            print("learn:{}/{},epoch:{}".format(step + 1, MAX_STEP, episode + 1))
            if total_step % 10000 == 9999 and total_step < 80000:
                MyNet.Agent.variance *= 0.8  # 每执行10000step减小噪声方差
                ''''''
                for p1 in MyNet.Agent.critic_1_optim.param_groups:  # 动态调整学习率，提升学习效果
                    p1['lr'] *= 0.3
                for p2 in MyNet.Agent.critic_2_optim.param_groups:
                    p2['lr'] *= 0.3
                for p3 in MyNet.Agent.actor_optim.param_groups:
                    p3['lr'] *= 0.3

        reward.append(episode_reward)
        time_end = time.perf_counter()
        time_span.append(time_end - time_start)
        print("============episode:{}/{}============".format(episode + 1, EPOCH))
    MyNet.Agent.save_td3()
    t2 = time.perf_counter()
    train_hour = int((t2 - t1)//3600)
    train_minute = int((t2 - t1)//60) - train_hour * 60
    train_second = int((t2 - t1) % 60)
    print('模型训练完毕！用时{:d}时{:d}分{:d}秒'.format(train_hour, train_minute, train_second))

    # 显示训练曲线
    plt.subplot(2, 1, 1)
    plt.plot(reward, linewidth=0.6, color='green')
    plt.subplot(2, 1, 2)
    plt.plot(time_span, linewidth=0.6, color='red')  # 时间性能分析
    plt.show()
    t2 = time.perf_counter()  # 显示训练曲线时程序会暂停，因此需要刷新一下时间

else:  # 非训练模式下，需从训练好的文件中读取数据
    MyNet.Agent.load_td3()
    t2 = time.perf_counter()
    print('模型加载完毕！用时 {:.3f} 秒'.format(t2 - t1))

MyNet.area_reset()
MyNet.Agent.variance = 1e-3
state_last_1 = []
for i in range(room_number):
    state_last_1.append(Ti_ini_list[i]/50)
    state_last_1.append(Tm_ini_list[i]/50)
    state_last_1.append(Ti_ref_list[i]/50)
state_last_1.append(To_list[0])

for t in range(time_length):
    Q_list, r, state_last_1 = MyNet.power_allocation(Ti_ref_list, To_list[int((t - 1) * dt) + start_day * 24], state_last_1)
    # 根据TD3算功率
    MyNet.Agent.learn()  # 在实际环境中，可以根据实际情况一步步来学习
    Q_1.append(Q_list[0])
    Q_2.append(Q_list[1])
    Ti_1_list.append(MyNet.room_list[0].ti)
    Ti_2_list.append(MyNet.room_list[1].ti)
    Tm_1_list.append(MyNet.room_list[0].tm)
    Tm_2_list.append(MyNet.room_list[1].tm)
    To_record.append(To_list[int((t - 1) * dt) + start_day * 24])
    plt.ion()
    plt.clf()
    plt.plot(Ti_1_list, linewidth=0.5, label='Ti1')
    plt.plot(Ti_2_list, linewidth=0.5, label='Ti2')
    plt.plot(To_record, linewidth=0.5, label='To')
    plt.pause(0.02)
    plt.ioff()

plt.ioff()
t3 = time.perf_counter()
print('数据处理完毕！用时 {:.3f} 秒'.format(t3 - t2))

#作图
Ti_set_1_line = Ti_set_1 * np.ones(time_length)
Ti_set_2_line = Ti_set_2 * np.ones(time_length)

fig = plt.figure()
ax1 = fig.add_subplot(5, 1, 1)
ax2 = fig.add_subplot(5, 1, 3)
ax3 = fig.add_subplot(5, 1, 5)

#第一张图绘制房间1的室温与墙壁温度
ax1.plot(Ti_1_list, linewidth=0.6, label='Ti1')
ax1.plot(Tm_1_list, linewidth=0.6, label='Tm1')
ax1.plot(To_record, linewidth=0.6, label='To')
ax1.plot(Ti_set_1_line, linewidth=0.3, linestyle='--')
ax1.legend()
ax1.set_title('Room1')
ax1.set_xlabel('Time/4sec')
ax1.set_ylabel('Temp/℃')

#第二张图绘制房间2的室温与墙壁温度
ax2.plot(Ti_2_list, linewidth=0.6, label='Ti2')
ax2.plot(Tm_2_list, linewidth=0.6, label='Tm2')
ax2.plot(To_record, linewidth=0.6, label='To')
ax2.plot(Ti_set_2_line, linewidth=0.3, linestyle='--')
ax2.legend()
ax2.set_title('Room2')
ax2.set_xlabel('Time/4sec')
ax2.set_ylabel('Temp/℃')

#第三张图绘制功率曲线，以电功率为基准
P_list = []
Qe1 = []
Qe2 = []
for i in range(time_length):
    P_list.append(abs(Q_1[i])/COP + abs(Q_2[i])/COP)
    Qe1.append(abs(Q_1[i])/COP)
    Qe2.append(abs(Q_2[i])/COP)
ax3.plot(Qe1, linewidth=0.8, label='Q1')
ax3.plot(Qe2, linewidth=0.8, label='Q2')
ax3.plot(P_list, linewidth=0.8, label='Q_total')
ax3.legend()
ax3.set_title('Electricity Power')
ax3.set_xlabel('Time/4sec')
ax3.set_ylabel('Power/W')

plt.show()
