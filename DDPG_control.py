"""
本代码为DDPG算法训练后仿真，先训练好模型，再进行仿真
训练模型的过程很耗时，如果想要让模型实现对全年温度控制的学习，还会出现内存不够用的问题
"""
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

dt = 1 / 60 / 15    #时间微元为4秒
COP = 3  #电功率热功率转化系数


"""
采用算法：DDPG（deep deterministic policy gradient，深度确定性策略梯度算法）
DDPG算法本身是建立在DQN基础上的，使用Actor输出动作，使用Critic输出q值，可以用来处理连续性问题
本项目预期功率属于连续值，同时状态量中的温度与regD功率也属于连续量，因此采用可以处理连续问题的DDPG算法
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
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 0.1)

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
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 0.1)

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
    def __init__(self, gamma, actor_lr, critic_lr, tau, capacity, batch_size, save_path):
        self.gamma = gamma  # 每次迭代奖励值的折扣系数
        self.actor_lr = actor_lr  # Actor网络的学习率
        self.critic_lr = critic_lr  # Critic网络的学习率
        self.tau = tau  # 网络软更新参数
        self.capacity = capacity  # 内存池容量
        self.batch_size = batch_size  # 每次更新的容量
        self.save_path = save_path

        self.s_dim = 4  # 状态维度有5个：Ti, Tm, To和Ti_set
        self.a_dim = 1  # 动作维度仅1个，即热功率

        # 神经网络的初始化，需要建立4个网络：Actor, Critic, Actor的目标网络， Critic的目标网络
        # 采用目标网络可以避免Q值的更新发生震荡
        self.actor = Actor(self.s_dim, 20, self.a_dim)
        self.actor.to(device)
        self.actor_target = Actor(self.s_dim, 20, self.a_dim)
        self.actor_target.to(device)
        self.critic = Critic(self.s_dim + self.a_dim, 64, self.a_dim)
        self.critic.to(device)
        self.critic_target = Critic(self.s_dim + self.a_dim, 64, self.a_dim)
        self.critic_target.to(device)
        # 神经网络优化器以及经验回放池的建立
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = np.array([0, 0, 0, 0, 0, 0, 0])
        # 初始化状态下将神经网络本体与目标网络进行同步
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

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
        if len(self.buffer)-1 < self.batch_size:  # 经验池中内容若小于批量更新数，则先不进行学习
            return

        #从经验回放池中随机采样，通过选取参数进行，便于直接从经验池中获取下一时刻的数据
        index = np.random.choice(len(self.buffer)-1, size=self.batch_size, replace=True)
        sample = self.buffer[index, :]
        sample_next = self.buffer[index+1, :]

        #采样的内容包含此时刻的动作、奖励与状态以及下一时刻的状态
        #由于buffer里面单条存储内容不包含下一时刻的s，因此r1与s1的表示方式也要改变，否则无法达到理想效果！
        s_now = sample[:, :self.s_dim]
        a_now = sample[:, self.s_dim:self.s_dim+self.a_dim]
        r_this = sample[:, self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        d_this = sample[:, self.s_dim+self.a_dim+1:self.s_dim+self.a_dim+2]
        s_next = sample_next[:, :self.s_dim]
        # 将状态与动作转化为pytorch可以处理的张量形式
        s_now = np.array(s_now)
        a_now = np.array(a_now)
        r_this = np.array(r_this)
        d_this = np.array(d_this)
        s_next = np.array(s_next)
        s_now = torch.tensor(s_now, dtype=torch.float).to(device)
        a_now = torch.tensor(a_now, dtype=torch.float).to(device)
        r_this = torch.tensor(r_this, dtype=torch.float).view(self.batch_size, -1).to(device)
        d_this = torch.tensor(d_this, dtype=torch.float).view(self.batch_size, -1).to(device)
        s_next = torch.tensor(s_next, dtype=torch.float).to(device)

        def critic_learn():
            # Critic网络的学习与更新
            a1 = self.actor_target(s_next).detach()  # 选出下一时刻最优化的动作

            y_true = r_this + self.gamma * self.critic_target(s_next, a1).detach() * d_this
            # 根据目标网络定义y值，作为更新网络的依据
            y_predict = self.critic(s_now, a_now)  # 从Critic网络中读取q值作为y预测值

            loss_fn = nn.MSELoss()  # 定义损失函数MSELoss： 输入x（模型预测输出）和目标y之间的均方误差标准
            #  通过使损失函数最小化寻找最优值以更新Critic函数
            loss = loss_fn(y_predict, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            # Actor网络的学习与更新
            loss = -torch.mean(self.critic(s_now, self.actor(s_now)))  # 利用采样后的策略梯度的均值计算损失函数，并更新Actor网络
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            # 神经网络软更新，使每次迭代都更新目标网络时，算法也能保持一定的稳定性
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        # 进行网络的学习与更新
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def load_ddpg(self):
        # 调用现成的模型文件，方便直接应用
        try:
            checkpoint = torch.load(self.save_path)
            self.actor.load_state_dict(checkpoint['actor_net_dict'])
            self.critic.load_state_dict(checkpoint['critic_net_dict'])
            self.actor_optim.load_state_dict(checkpoint['actor_net_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_net_dict'])
        except Exception as e:
            print(e)

    def save_ddpg(self):
        # 保存训练模型，在训练完以后便于直接调用训练文件
        torch.save(
            {
                'actor_net_dict': self.actor.state_dict(),
                'critic_net_dict': self.critic.state_dict(),
                'actor_optim_dict': self.actor_optim.state_dict(),
                'critic_optim_dict': self.critic_optim.state_dict()
            }, self.save_path)


class ETPRoom:
    """
    该类用于单个房间的模型构建，便于处理多个房间的问题
    参数：
    ki, kl, ko, ca, cm ： 房间的ETP模型参数
    ti_ini, tm_ini ： 房间的初始温度
    gamma, actor_lr, critic_lr, tau, capacity, batch_size： 强化学习参数
    """
    def __init__(self, ki, kl, ko, ca, cm, ti_ini, tm_ini,
                 gamma, actor_lr, critic_lr, tau, capacity, batch_size, save_pth, power, norm_variance=0.1):
        #参数的初始化
        self.ki = ki
        self.kl = kl
        self.ko = ko
        self.ca = ca
        self.cm = cm
        self.ti = ti_ini
        self.tm = tm_ini
        self.done = False
        self.ti_ini = ti_ini
        self.tm_ini = tm_ini
        self.power_limit = power * 2
        self.Agent = Agent(gamma, actor_lr, critic_lr, tau, capacity, batch_size, save_pth)
        self.variance = norm_variance

    def reset(self):
        self.ti = self.ti_ini
        self.tm = self.tm_ini
        self.done = False

    def cal_temp(self, to_last, q_last):
        # 房间温度的预测
        self.ti = self.ti + dt * 3600 * 15 * (
                self.ki * (self.tm - self.ti) + self.kl * (to_last - self.ti) + q_last) / self.ca
        self.tm = self.tm + dt * 3600 * 15 * (
                self.ki * (self.ti - self.tm) + self.ko * (to_last - self.tm)) / self.cm
        return self.ti, self.tm

    def ddpg_controller(self, t_o, ti_ref):
        # 利用DDPG算法调整房间温度
        s00 = np.array([self.ti/50, self.tm/50, t_o/50, ti_ref/50])
        # 先规定当前步骤的状态，需要注意归一化
        a0 = self.Agent.act(s00) + np.random.normal(0, self.variance, size=1)
        # 选择最优动作（功率），需要注意归一化与噪声探索
        a00 = (float(a0)-0.5) * self.power_limit * COP * 2
        # 根据归一化功率反推实际热功率，带有一些探索量
        self.ti, self.tm = self.cal_temp(t_o, a00)  # 根据ETP模型计算下一时刻的温度并更新状态
        r1 = 0.01 - abs(self.ti - ti_ref)/100 - abs(a00)/self.power_limit/10000
        # 规定奖励，该奖励取决于温度差与regD跟踪效果，使用归一化数据，保证r不会太大
        if not self.done:
            d0 = 1
        else:
            d0 = 0
        trans0 = np.append(s00, a0)
        trans1 = np.append(trans0, r1)
        trans = np.append(trans1, d0)
        self.Agent.put(trans)  # 往经验回放池中存储动作

        return abs(a00/COP), r1


# 文件读取
t0 = time.perf_counter()
Shanghai_data = pd.read_excel(io='./CHN_Shanghai.Shanghai.583670_IWEC.xlsx')  # 温度数据
To_list = []
for i in Shanghai_data.index.values:
    row_data = Shanghai_data.loc[i, ['Dry Bulb Temperature']].to_dict()
    row_data_0 = list(row_data.values())
    To_list.append(row_data_0[0])
RegD_data = pd.read_excel(io='./reg-data-external-may-2014.xlsx', sheet_name=0)  # RegD信号数据
RegD_list = []
for i in RegD_data.index.values:
    reg_row = RegD_data.loc[i, pd.to_datetime(['2014/5/4'])].to_dict()
    reg_row_0 = list(reg_row.values())
    RegD_list.append(reg_row_0[0])
t1 = time.perf_counter()
print('文件读取完毕！用时 {:.3f} 秒'.format(t1 - t0))

To_record = []
start_day = 0
base_power = 40000

Ti_ini = To_list[start_day * 24]
Tm_ini = To_list[start_day * 24]
Ti_list = [To_list[start_day * 24]]  # 从5月4日开始进行仿真；数据的年份并不匹配，但是仍然具有合理性
Tm_list = [To_list[start_day * 24]]

Q_1 = []
Q_reg_list = []

time_length = 60 * 16 * 15  # 总时长，设为1d，每4秒为1个时段
Ti_set = 25  # 房间设定温度

# 强化学习参数
GAMMA = 0.99
ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
TAU = 0.001
CAPACITY = 235000
BATCH_SIZE = 32
EPOCH = 200
MAX_STEP = 60 * 8 * 15
TRAIN_MODE = True
SAVE_PTH = "DDPG_control_summer.pth"
INI_VARIANCE = 0.1

MyRoom = ETPRoom(ki_1, kl_1, ko_1, Ca_1, Cm_1, Ti_ini, Tm_ini,
                 GAMMA, ACTOR_LR, CRITIC_LR, TAU, CAPACITY, BATCH_SIZE, SAVE_PTH, base_power, INI_VARIANCE)  # 房间初始化

if TRAIN_MODE:  # 训练模式下，直接开启训练并生成训练模型
    reward = []
    time_span = []
    total_step = 0
    for episode in range(EPOCH):  # 训练过程
        # Ti_train_set = random.randint(180, 300) / 10  # 训练时随机设定目标温度，从而使模型有更好的普适性
        Ti_train_set = 25  # 老老实实先练好当前的再说
        episode_reward = 0
        MyRoom.reset()
        time_start = time.perf_counter()
        for step in range(MAX_STEP):
            '''
            if step % (60 * 15) == 0 and step != 0:
                # 训练时每1h更新一次设定温度，保证训练效果的同时减少需要的迭代次数
                Ti_train_set = random.randint(180, 300) / 10
            '''
            total_step += 1
            if step == MAX_STEP - 1:
                MyRoom.done = True
            Q_room, r = MyRoom.ddpg_controller(To_list[(step - 1) // 60 // 15], Ti_train_set)
            episode_reward += r
            if step % 10 == 9:
                MyRoom.Agent.learn()  # 每模拟10step学习一次，略微提升效率
                print("learn:{}/{}".format(step+1, MAX_STEP))
            if total_step % 200000 == 199999 and total_step < 800000:
                MyRoom.variance /= 10  # 每执行200000step减小噪声方差
                for p1 in MyRoom.Agent.critic_optim.param_groups:  # 动态调整学习率，提升学习效果
                    p1['lr'] *= 0.1
                for p2 in MyRoom.Agent.actor_optim.param_groups:
                    p2['lr'] *= 0.1

        reward.append(episode_reward)
        time_end = time.perf_counter()
        time_span.append(time_end-time_start)
        print("============episode:{}/{}============".format(episode+1, EPOCH))
    MyRoom.Agent.save_ddpg()
    t2 = time.perf_counter()
    print('模型训练完毕！用时 {:.3f} 秒'.format(t2 - t1))
    # 显示训练曲线
    plt.subplot(2, 1, 1)
    plt.plot(reward, linewidth=0.6, color='green')
    plt.subplot(2, 1, 2)
    plt.plot(time_span, linewidth=0.6, color='red')  #时间性能分析
    plt.show()
    t2 = time.perf_counter()  # 显示训练曲线时程序会暂停，因此需要刷新一下时间

else:  # 非训练模式下，需从训练好的文件中读取数据
    MyRoom.Agent.load_ddpg()
    t2 = time.perf_counter()
    print('模型加载完毕！用时 {:.3f} 秒'.format(t2 - t1))

# 在拥有完成训练的模型的基础上进行仿真
MyRoom.reset()
MyRoom.variance = 0  # 训练好模型以后进行场景应用时，就不需要动作的探索了
for t in range(time_length):
    Q_reg = base_power / COP + 0.5 * base_power / COP * RegD_list[(t - 1) % 21600]  # 规定RegD信号
    Q_reg_list.append(Q_reg)
    Q_room, r = MyRoom.ddpg_controller(To_list[(t - 1) // 60 // 15 + start_day * 24], Ti_set)  # 根据DDPG算功率
    MyRoom.Agent.learn()  # 在实际环境中，可以根据实际情况一步步来学习
    Q_1.append(Q_room)
    Ti_list.append(MyRoom.ti)
    Tm_list.append(MyRoom.tm)
    To_record.append(To_list[(t - 1) // 60 // 15 + start_day * 24])

t3 = time.perf_counter()
print('数据处理完毕！用时 {:.3f} 秒'.format(t3 - t2))

# 作图部分
Ti_set_line = np.ones(time_length) * Ti_set

# 设置画布
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 3)

# 第一张图绘制房间温度变化
ax1.plot(Ti_list, linewidth=0.5, label='Ti')
ax1.plot(Tm_list, linewidth=0.5, label='Tm')
ax1.plot(To_record, linewidth=0.5, label='To')
ax1.plot(Ti_set_line, linewidth=0.1, linestyle='--')
ax1.legend()
ax1.set_title('Room_Temp')
ax1.set_xlabel('Time/4sec')
ax1.set_ylabel('Temp/℃')

# 第二张图绘制功率变化
P_list = []
for ele in range(time_length):
    P_list.append(abs(Q_1[ele]) / COP)
ax2.plot(P_list, linewidth=0.8, label='Q_total')
ax2.plot(Q_reg_list, linewidth=0.8, label='Q_reg')
ax2.legend()
ax2.set_title('Electricity Power')
ax2.set_xlabel('Time/4sec')
ax2.set_ylabel('Power/W')

# 显示仿真结果
plt.show()
