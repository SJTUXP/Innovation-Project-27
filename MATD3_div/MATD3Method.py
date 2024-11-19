import numpy as np
import torch
import torch.nn as nn
import MATD3Agent
import collections
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MATD3:
    def __init__(self, room_num, actor_lr, critic_lr, gamma, tau, save_path,
                 norm_variance=0.1, noise_scale=0.2, delay_time=2):
        self.agents = []  #看来不是智能体存储方式的问题，这样存储实际上没问题
        for i in range(room_num):
            self.agents.append(MATD3Agent.Agent(actor_lr, critic_lr, room_num * 4))
        self.gamma = gamma  # 每次迭代奖励值的折扣系数
        self.tau = tau  # 网络软更新参数
        self.save_path = save_path  # 文件保存路径
        self.variance = norm_variance  # 高斯噪声方差
        self.update_time = 0  # 更新次数，用于Actor的延迟更新策略
        self.delay_time = delay_time  # 延迟更新时间，用于Actor的延迟更新策略
        self.room_num = room_num
        self.noise_scale = noise_scale  # 噪声规模，限制探索过程中产生的动作幅度

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.actor_target for agt in self.agents]

    def act_add_noise(self, act):
        noise = torch.tensor(np.clip(np.random.normal(0, self.variance, size=1),
                                     -self.noise_scale / 2, self.noise_scale / 2), dtype=torch.float).to(device)
        return torch.clamp(act.detach() + noise, -1, 1)

    def take_actions(self, states):
        states = [torch.tensor(np.array([states[i]]), dtype=torch.float, device=device) for i in range(self.room_num)]
        return [np.clip(agent.act(state) + np.clip(np.random.normal(0, self.variance, size=1),
                                                   -self.noise_scale / 2, self.noise_scale / 2), -1, 1)
                for agent, state in zip(self.agents, states)]

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor_target, agt.actor, self.tau)
            agt.soft_update(agt.critic_1_target, agt.critic_1, self.tau)
            agt.soft_update(agt.critic_2_target, agt.critic_2, self.tau)

    @staticmethod
    def stack_array(x):
        rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
        return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]

    def learn(self, samples):
        samples = [self.stack_array(x) for x in samples]
        s_now, act, rew, s_next, done = samples
        loss_fn = nn.MSELoss()  # 定义损失函数MSELoss： 输入x（模型预测输出）和目标y之间的均方误差标准
        self.update_time += 1
        for i_agent, cur_agent in enumerate(self.agents):
            cur_agent.critic_1_optim.zero_grad()
            cur_agent.critic_2_optim.zero_grad()

            all_target_act = [self.act_add_noise(pi(_s_next))
                              for pi, _s_next in zip(self.target_policies, s_next)]
            q_true = torch.min(cur_agent.critic_1_target(s_next, all_target_act).detach(),
                               cur_agent.critic_2_target(s_next, all_target_act).detach())
            y_true = rew[i_agent].view(-1, 1) + self.gamma * done[i_agent] * q_true

            y_1_predict = cur_agent.critic_1(s_now, act)
            y_2_predict = cur_agent.critic_2(s_now, act)

            #  通过使损失函数最小化寻找最优值以更新Critic函数
            loss_1 = loss_fn(y_1_predict, y_true)
            loss_2 = loss_fn(y_2_predict, y_true)
            loss = loss_1 + loss_2
            loss.backward()
            cur_agent.critic_1_optim.step()
            cur_agent.critic_2_optim.step()

            # Actor网络的学习与更新
            if self.update_time % self.delay_time == 0:
                cur_agent.actor_optim.zero_grad()
                all_act = [pi(_s) + torch.tensor(np.clip(np.random.normal(0, self.variance, size=1),
                                                         -self.noise_scale / 2, self.noise_scale / 2),
                                                 dtype=torch.float).to(device)
                           for pi, _s in zip(self.policies, s_now)]
                all_act[i_agent] = cur_agent.actor(s_now[i_agent])
                cur_act = cur_agent.actor(s_now[i_agent])
                loss_act = -torch.mean(cur_agent.critic_1(s_now, all_act)) + torch.mean(cur_act ** 2) * 1e-6
                '''
                代码出现问题的地方
                最好的方法就是简化获取变量的操作，使得loss可以直接回传到所需的神经网络之中
                '''
                # 利用采样后的策略梯度的均值计算损失函数，并更新Actor网络
                # 对于actor来说，其实并不在乎Q值是否会被高估，他的任务只是不断做梯度上升，寻找这条最大的Q值。
                # 随着更新的进行，Q1和Q2两个网络将会变得越来越像。所以用Q1还是Q2，还是两者都用，对于actor的问题不大。
                loss_act.backward()
                cur_agent.actor_optim.step()

                self.update_all_targets()

    def load_td3(self):
        # 调用现成的模型文件，方便直接应用
        try:
            for i, agt_i in enumerate(self.agents):
                checkpoint = torch.load(self.save_path + "_agent{:d}".format(i + 1) + ".pth")
                agt_i.actor.load_state_dict(checkpoint['actor_net_dict'])
                agt_i.critic_1.load_state_dict(checkpoint['critic_1_net_dict'])
                agt_i.critic_2.load_state_dict(checkpoint['critic_2_net_dict'])
                agt_i.actor_optim.load_state_dict(checkpoint['actor_optim_dict'])
                agt_i.critic_1_optim.load_state_dict(checkpoint['critic_1_optim_dict'])
                agt_i.critic_2_optim.load_state_dict(checkpoint['critic_2_optim_dict'])

        except Exception as e:
            print(e)

    def save_td3(self):
        # 保存训练模型，在训练完以后便于直接调用训练文件
        i_agt = 0
        for agt in self.agents:
            i_agt += 1
            torch.save(
                {
                    'actor_net_dict': agt.actor.state_dict(),
                    'critic_1_net_dict': agt.critic_1.state_dict(),
                    'critic_2_net_dict': agt.critic_2.state_dict(),
                    'actor_optim_dict': agt.actor_optim.state_dict(),
                    'critic_1_optim_dict': agt.critic_1_optim.state_dict(),
                    'critic_2_optim_dict': agt.critic_2_optim.state_dict(),
                }, self.save_path + "_agent{:d}".format(i_agt) + ".pth")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def sample(self, batch_size):
        index = random.sample([i for i in range(len(self.buffer) - 1)], batch_size)
        transitions = [self.buffer[i] for i in index]
        transitions_next = [self.buffer[i + 1] for i in index]
        state, action, reward, done = zip(*transitions)
        state_next, _, _, _ = zip(*transitions_next)
        return np.array(state), action, reward, np.array(state_next), done

    def size(self):
        return len(self.buffer)


"""
# 目前自制的经验回放池还有问题
class MATD3_Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def put(self, *transition):
        # 将新的状态、动作与奖励存储进经验回放池中
        # 如果经验回放池已满，则将最早的经验记录清除
        self.buffer.append(transition)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            return
        index = np.random.choice(len(self.buffer) - 1, size=batch_size, replace=True)
        sample1 = np.array(self.buffer)[index][0]
        sample_next = np.array(self.buffer)[index + 1][0]

        s_now = sample1[:, 0]
        a_now = sample1[:, 1]
        r_this = sample1[:, 2]
        d_this = sample1[:, 3]
        s_next = sample_next[:, 0]
        # 将状态与动作转化为pytorch可以处理的张量形式
        s_now = np.array(s_now)
        a_now = np.array(a_now)
        r_this = np.array(r_this)
        d_this = np.array(d_this)
        s_next = np.array(s_next)
        s_now = torch.tensor(s_now, dtype=torch.float).to(device)
        a_now = torch.tensor(a_now, dtype=torch.float).to(device)
        r_this = torch.tensor(r_this, dtype=torch.float).view(batch_size, -1).to(device)
        d_this = torch.tensor(d_this, dtype=torch.float).view(batch_size, -1).to(device)
        s_next = torch.tensor(s_next, dtype=torch.float).to(device)

        return s_now, a_now, r_this, s_next, d_this
"""
