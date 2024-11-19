import torch
import torch.nn as nn
import torch.nn.functional as fc


# Actor网络，用于输出动作
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 网络初始化：输入层、隐藏层、输出层
        # 输入层代表输入的状态维度，输出层代表动作的维度，隐藏层则用于建立连接
        super(Actor, self).__init__()
        # 网络连接层初始化，这里采用2个线性层
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        # 前向函数设立，将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止
        # fc.relu()函数：relu(x) = max(0, x)
        # torch.tanh()函数：(e**x-e**(-x))/(e**x+e**(-x))
        x = fc.elu(self.linear1(s), 0.001)
        x = torch.tanh(self.linear2(x))
        return x


# Critic网络，用于输出q值，从而影响动作的选择
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 网络初始化：输入层、隐藏层、输出层
        # 输入层代表输入的状态维度，输出层代表动作的维度，隐藏层则用于建立连接
        super().__init__()
        # 网络连接层初始化，这里采用3个线性层
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.linear3 = nn.Linear(int(hidden_size / 2), output_size)

    def forward(self, s, a):
        # 前向函数设立，将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止
        # fc.relu()函数：relu(x) = max(0, x)
        x = torch.cat((*s, *a), 1)
        x = fc.elu(self.linear1(x), 0.001)
        x = fc.elu(self.linear2(x), 0.001)
        x = self.linear3(x)
        return x
