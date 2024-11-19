import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import time
import platform
print('系统:', platform.system())

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

'''ki_list = [ki_1]
kl_list = [kl_1]
ko_list = [ko_1]
ca_list = [Ca_1]
cm_list = [Cm_1]'''
ki_list = [ki_1, ki_2]
kl_list = [kl_1, kl_2]
ko_list = [ko_1, ko_2]
ca_list = [Ca_1, Ca_2]
cm_list = [Cm_1, Cm_2]

# 此处将房间的参数整理为列表，方便网络类进行处理与初始化。若拓展至更多房间，该参数需要进行修改

dt = 1 / 60 / 15  # 时间微元为4秒
N = 3  # MPC预测步长
COP = 3  # 冷热功率换算系数


class AreaNet:
    """
    该类为园区网络的模型构建，用于处理各个房间之间的交互数据
    参数：
    room_num：房间数量
    ki_lst, kl_lst, ko_lst, ca_lst, cm_lst：房间参数列表
    step：MPC预测步长
    para_q_lst, para_w_lst：MPC控制参数列表
    ti_ini_lst, tm_ini_lst：房间初始温度参数列表
    """

    def __init__(self, room_num, ki_lst, kl_lst, ko_lst, ca_lst, cm_lst, step, para_q_lst, para_w_lst, ti_ini_lst,
                 tm_ini_lst):
        if np.all([len(ki_lst), len(kl_lst), len(ko_lst), len(ca_lst), len(cm_lst), len(para_q_lst), len(para_w_lst),
                   len(ti_ini_lst), len(tm_ini_lst)] == room_num):
            print([len(ki_lst), len(kl_lst), len(ko_lst), len(ca_lst), len(cm_lst), len(para_q_lst), len(para_w_lst),
                   len(ti_ini_lst), len(tm_ini_lst)])
            print(room_num)
            raise ValueError(
                "Room number is not matched with room parameter lists!"
            )
        # 先判断房间的数量是否与参数匹配，若不匹配则报错，避免因为参数不匹配导致后续的运算错误

        self.room_n = room_num
        self.room_list = [
            self.ETPRoom(ki_lst[room_index], kl_lst[room_index], ko_lst[room_index], ca_lst[room_index],
                         cm_lst[room_index], step, para_q_lst[room_index], para_w_lst[room_index],
                         ti_ini_lst[room_index], tm_ini_lst[room_index])
            for room_index in range(room_num)]  # 房间列表
        for room in self.room_list:
            room.cal_matrix()  # 计算房间的MPC控制矩阵，若无此步MPC控制将失效
        self.mpc_state = np.ones(room_num)  # 是否为mpc控制的状态列表，初始认为全部进行mpc控制

        self.step = step
        self.ti_list = ti_ini_lst
        self.tm_list = tm_ini_lst
        self.q_list = np.zeros(room_num)  # 园区各个房间的状态列表

    class ETPRoom:
        """
        该类用于单个房间的模型构建，便于处理多个房间的问题
        参数：
        ki, kl, ko, ca, cm ： 房间的ETP模型参数
        n : MPC预测步长
        para_q, para_w : MPC控制矩阵的系数
        """

        def __init__(self, ki, kl, ko, ca, cm, n_step, para_q, para_w, ti_ini, tm_ini):
            # 参数的初始化
            self.ki = ki
            self.kl = kl
            self.ko = ko
            self.ca = ca
            self.cm = cm
            self.n = n_step
            self.q_m = para_q * np.eye(2 * n_step)
            self.w = para_w * np.eye(n_step)
            self.psi = np.zeros((2 * self.n, 2))  # 状态预测量系数矩阵
            self.theta = np.zeros((2 * self.n, self.n))  # 控制预测量系数矩阵
            self.beta = np.zeros((2 * self.n, self.n))  # 环境预测量系数矩阵
            self.ti = ti_ini
            self.tm = tm_ini

        def cal_temp(self, to_last, q_last):
            # 房间温度的预测
            self.ti = self.ti + dt * 3600 * 15 * (
                    self.ki * (self.tm - self.ti) + self.kl * (to_last - self.ti) + q_last) / self.ca
            self.tm = self.tm + dt * 3600 * 15 * (
                    self.ki * (self.ti - self.tm) + self.ko * (to_last - self.tm)) / self.cm
            return self.ti, self.tm

        def cal_matrix(self):
            # MPC控制的矩阵计算，计算出的矩阵可以通过房间类直接传递给mpc_controller()函数
            A = np.array([[-(self.ki + self.kl) * dt / self.ca + 1, self.ki * dt / self.ca],
                          [self.ki * dt / self.cm, -(self.ki + self.ko) * dt / self.cm + 1]])  # 状态量系数矩阵
            B = np.array([[self.kl * dt / self.ca], [0]])  # 控制量系数矩阵
            C = np.array([[self.kl * dt / self.ca], [self.ko * dt / self.cm]])  # 环境量参数矩阵
            a0 = A
            for row in range(self.n):
                self.psi[row * 2][0] = a0[0][0]
                self.psi[row * 2][1] = a0[0][1]
                self.psi[row * 2 + 1][0] = a0[1][0]
                self.psi[row * 2 + 1][1] = a0[1][1]
                a0 = np.matmul(a0, A)
            for row in range(self.n):
                for col in range(row + 1):
                    a = np.eye(2)
                    if row - col > 0:
                        for k in range(row - col):
                            a = np.matmul(a, A)
                    ele = np.matmul(a, B)
                    ele2 = np.matmul(a, C)
                    self.theta[row * 2][col] = ele[0][0]
                    self.theta[row * 2 + 1][col] = ele[1][0]
                    self.beta[row * 2][col] = ele2[0][0]
                    self.beta[row * 2 + 1][col] = ele2[1][0]

            # return self.psi, self.theta, self.beta

        def mpc_controller(self, ti_ref, p_reg_k, t_o):
            # MPC控制的实现，在计算温度时取第一个预测功率值作为下一时刻的功率，在进行功率分配时需要考虑所有预测量
            rk = np.zeros((2 * self.n, 1))  # 参考值矩阵创建
            otk = np.ones((self.n, 1)) * t_o  # 室外温度预测矩阵，短时间内室外温度变化不会太大，因此可以使用同一量
            for i0 in range(self.n):
                rk[2 * i0][0] = ti_ref
                rk[2 * i0 + 1][0] = self.tm  # 参考值奇数项元素为设定温度， 偶数项为墙壁温度
            xk = [[self.ti], [self.tm]]

            '''
            g2 = 1
            g1 = 1  # 惩罚系数
            '''
            g2 = (self.ti - ti_ref) ** 2 * 0.99 / ((self.ti - ti_ref) ** 2 + 1) + 0.01
            g1 = 1 / ((self.ti - ti_ref) ** 2 + 1)  # 惩罚系数

            # 将regD电功率信号转化为热功率
            reg_heat = []
            for reg_i in range(len(p_reg_k)):
                if self.ti > ti_ref:
                    reg_heat.append(- p_reg_k[reg_i] * COP)
                else:
                    reg_heat.append(p_reg_k[reg_i] * COP)

            E = np.matmul(self.psi, xk) - rk + np.matmul(self.beta, otk)
            # 此处进行过一次修改，补充了之前缺失的室外温度项，所需矩阵也在前面部分补充
            H = 2 * (g1 * np.matmul(np.matmul(self.theta.transpose(), self.q_m), self.theta) + g2 * self.w)
            f = 2 * (g1 * np.matmul(np.matmul(E.transpose(), self.q_m), self.theta).transpose()
                     - g2 * np.matmul(reg_heat, self.w).reshape(self.n, 1))
            # 注意：修改目标函数以后该处代码极易出错！regD信号需要乘以COP，还要注意制冷制热的问题（这里直接使用符号函数），否则控制效果不理想！
            # 问题定位：np.matmul(reg_k * COP * np.sign(ti_ref-self.ti), self.w)
            # 第一代直接用regD，只能实现总功率和regD信号单调性一致：np.matmul(reg_k, self.w)
            # 第二代乘上了COP，有一定跟踪效果，但是温度控制不完全理想：np.matmul(reg_k * COP, self.w)
            # 第三代再乘上了温度差的符号函数来实现制冷制热的切换，大大改善了温度跟踪效果
            u_k_list = solve_qp(H, f, solver='cvxopt')

            # u_k = u_k_list[0]
            # 这句代码仅在单房间使用，返回值需要修改

            return u_k_list

    def temp_update(self, to_last):
        # 根据ETP模型算出每个房间的室温与墙壁温度
        for room_i in range(self.room_n):
            self.ti_list[room_i], self.tm_list[room_i] = self.room_list[room_i].cal_temp(to_last, self.q_list[room_i])
        return self.ti_list, self.tm_list

    def power_allocation(self, p_reg, u_k_last, ti_ref_list, t_o):
        # 用于计算各个房间的功率并且进行分配，分配依据是拉格朗日乘子法
        reg_list = []  #准备好regD信号功率分配列表，便于后续的功率分配

        # 拉格朗日乘子法的矩阵计算
        gu_sum = np.zeros(self.step)
        w_sum = np.zeros((self.step, self.step))

        for room_i in range(self.room_n):
            gu = 1 / ((self.room_list[room_i].ti - ti_ref_list[room_i]) ** 2 + 1) * abs(u_k_last[room_i]) / COP
            gu_sum += gu
            w_sum += np.linalg.inv(self.room_list[room_i].w)
        w_new = np.linalg.inv(w_sum)
        lam = -2 * np.matmul((p_reg - gu_sum), w_new)  #拉格朗日乘子

        # 依据矩阵计算结果进行regD功率分配
        for room_i in range(self.room_n):
            reg_list.append(1 / ((self.room_list[room_i].ti - ti_ref_list[room_i]) ** 2 + 1)
                            * abs(u_k_last[room_i]) / COP
                            - 0.5 * np.matmul(lam, np.linalg.inv(self.room_list[room_i].w)))

        #对实际功率进行调控
        u_k_list = []
        for room_i in range(self.room_n):
            u_k = self.room_list[room_i].mpc_controller(ti_ref_list[room_i], reg_list[room_i], t_o)
            self.q_list[room_i] = u_k[0]
            u_k_list.append(u_k)

        return self.q_list, u_k_list, reg_list


# 文件读取
t0 = time.perf_counter()
Q_1 = [0]
Q_2 = [0]
Q_reg_list = []
Shanghai_data = pd.read_excel(io='./CHN_Shanghai.Shanghai.583670_IWEC.xlsx')  #温度数据
To_list = []
for i in Shanghai_data.index.values:
    row_data = Shanghai_data.loc[i, ['Dry Bulb Temperature']].to_dict()
    row_data_0 = list(row_data.values())
    To_list.append(row_data_0[0])
RegD_data = pd.read_excel(io='./reg-data-external-may-2014.xlsx', sheet_name=0)  #RegD信号数据
RegD_list = []
for i in RegD_data.index.values:
    reg_row = RegD_data.loc[i, pd.to_datetime(['2014/5/4'])].to_dict()
    reg_row_0 = list(reg_row.values())
    RegD_list.append(reg_row_0[0])
t1 = time.perf_counter()
print('文件读取完毕！用时 {:.3f} 秒'.format(t1-t0))

# 参数设定
To_record = []
start_day = 0
'''
Ti_ini_list = [20, 20]
Tm_ini_list = [20, 20]
Ti_1_list = [20]  #从5月4日开始进行仿真；数据的年份并不匹配，但是仍然具有合理性
Ti_2_list = [20]
Tm_1_list = [20]
Tm_2_list = [20]
'''

Ti_ini_list = [To_list[start_day*24], To_list[start_day*24]]
Tm_ini_list = [To_list[start_day*24], To_list[start_day*24]]
# Ti_ini_list = [To_list[start_day*24]]
# Tm_ini_list = [To_list[start_day*24]]
Ti_1_list = [To_list[start_day*24]]  #从5月4日开始进行仿真；数据的年份并不匹配，但是仍然具有合理性
Ti_2_list = [To_list[start_day*24]]
Tm_1_list = [To_list[start_day*24]]
Tm_2_list = [To_list[start_day*24]]

time_length = 60 * 8 * 15  #总时长，设为1d，每4秒为1个时段
t = 1
Ti_set_1 = 25  #房间1设定温度
Ti_set_2 = 24  #房间2设定温度
Ti_ref_list = [Ti_set_1, Ti_set_2]
# Ti_ref_list = [Ti_set_1]
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

q_m_1 = 1
w_1 = 5e-10
q_m_2 = 1
w_2 = 6e-10  #MPC矩阵控制参数
para_q_list = [q_m_1, q_m_2]
para_w_list = [w_1, w_2]
'''para_q_list = [q_m_1]
para_w_list = [w_1]'''
U_k_list = np.zeros((room_number, N))

MyNet = AreaNet(room_number, ki_list, kl_list, ko_list, ca_list, cm_list,
                N, para_q_list, para_w_list, Ti_ini_list, Tm_ini_list)
# 初始化园区及房间

while t < time_length:
    # 更新园区内各个房间的温度
    Ti_list, Tm_list = MyNet.temp_update(To_list[(t - 1) // 60 // 15 + start_day*24])
    Ti_1_list.append(Ti_list[0])
    Tm_1_list.append(Tm_list[0])
    Ti_2_list.append(Ti_list[1])
    Tm_2_list.append(Tm_list[1])

    # 分别规定2个房间的RegD信号，其基准信号来自etp模型的论文；冬季环境所需基准功率有所提升
    P_reg = [base_power + 0.5 * base_power * RegD_list[(t - 1 + i) % 21600] for i in range(N)]
    P_reg = np.array(P_reg)
    Q_reg_list.append(P_reg[0])

    # 更新园区内功率情况
    Q_list, U_k_list, Reg = MyNet.power_allocation(P_reg, U_k_list, Ti_ref_list,
                                                   To_list[(t - 1) // 60 // 15 + start_day*24])
    Q_1.append(Q_list[0])
    Q_2.append(Q_list[1])

    To_record.append(To_list[(t - 1) // 60 // 15 + start_day * 24])
    t += 1

t2 = time.perf_counter()
print('数据处理完毕！用时 {:.3f} 秒'.format(t2-t1))

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
ax3.plot(Q_reg_list, linewidth=0.8, label='Q_reg')
ax3.legend()
ax3.set_title('Electricity Power')
ax3.set_xlabel('Time/4sec')
ax3.set_ylabel('Power/W')

plt.show()

'''fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 3)

#第一张图绘制房间1的室温与墙壁温度
ax1.plot(Ti_1_list, linewidth=0.6, label='Ti1')
ax1.plot(Tm_1_list, linewidth=0.6, label='Tm1')
ax1.plot(To_record, linewidth=0.6, label='To')
ax1.plot(Ti_set_1_line, linewidth=0.3, linestyle='--')
ax1.legend()
ax1.set_title('Room1')
ax1.set_xlabel('Time/4sec')
ax1.set_ylabel('Temp/℃')

#第二张图绘制功率曲线，以电功率为基准
P_list = []
Qe1 = []
for i in range(time_length):
    P_list.append(abs(Q_1[i])/COP)
ax2.plot(P_list, linewidth=0.8, label='Q_total')
ax2.plot(Q_reg_list, linewidth=0.8, label='Q_reg')
ax2.legend()
ax2.set_title('Electricity Power')
ax2.set_xlabel('Time/4sec')
ax2.set_ylabel('Power/W')

plt.show()'''
