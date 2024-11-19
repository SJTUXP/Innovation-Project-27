import numpy as np
import torch
import torch.optim as optim
import ACNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agent:
    def __init__(self, actor_lr, critic_lr, critic_input_dim):
        self.actor_lr = actor_lr  # Actor网络的学习率
        self.critic_lr = critic_lr  # Critic网络的学习率

        self.s_dim = 3  # 每个房间每一步的状态维度有2个：Ti和Ti_ref，整个园区额外有1个状态: To
        self.a_dim = 1  # 每个房间的动作维度仅1个，即热功率

        # 神经网络的初始化，需要建立6个网络：Actor, Critic_1, Critic_2, Actor的目标网络， Critic_1的目标网络, Critic_2的目标网络
        # Critic采用双重网络的目的是选择q值预测值更小的输出，减小Q值预测值被高估的问题
        # 采用目标网络可以避免Q值的更新发生震荡
        self.actor = ACNet.Actor(self.s_dim, 64, self.a_dim)
        self.actor.to(device)
        self.actor_target = ACNet.Actor(self.s_dim, 64, self.a_dim)
        self.actor_target.to(device)
        self.critic_1 = ACNet.Critic(critic_input_dim, 128, self.a_dim)
        self.critic_1.to(device)
        self.critic_2 = ACNet.Critic(critic_input_dim, 144, self.a_dim)
        self.critic_2.to(device)
        self.critic_1_target = ACNet.Critic(critic_input_dim, 128, self.a_dim)
        self.critic_1_target.to(device)
        self.critic_2_target = ACNet.Critic(critic_input_dim, 144, self.a_dim)
        self.critic_2_target.to(device)
        # 神经网络优化器的建立
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optim = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optim = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        # 初始化状态下将神经网络本体与目标网络进行同步
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def act(self, state):
        # 根据智能体目前的状态选择最优的动作
        action = self.actor(state).squeeze(0).detach().cpu().numpy()  # 借助Actor网络选择最优动作
        return action.astype(np.float32)

    @staticmethod
    def soft_update(net_target, net, tau):
        # 神经网络软更新，使每次迭代都更新目标网络时，算法也能保持一定的稳定性
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
