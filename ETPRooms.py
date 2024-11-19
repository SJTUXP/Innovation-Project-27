import numpy as np
import MATD3Method

dt = 1 / 60 / 15  # 时间微元为4秒


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

    def __init__(self, room_num, ki_lst, kl_lst, ko_lst, ca_lst, cm_lst, ti_ini_lst, center_power_list,
                 gamma, actor_lr, critic_lr, tau, capacity, save_pth,
                 norm_variance=0.1, noise_scale=0.2, delay=2):
        if np.all([len(ki_lst), len(kl_lst), len(ko_lst), len(ca_lst), len(cm_lst),
                   len(ti_ini_lst), len(center_power_list)] == room_num):
            print([len(ki_lst), len(kl_lst), len(ko_lst), len(ca_lst), len(cm_lst),
                   len(ti_ini_lst), len(center_power_list)])
            print(room_num)
            raise ValueError(
                "Room number is not matched with room parameter lists!"
            )
        # 先判断房间的数量是否与参数匹配，若不匹配则报错，避免因为参数不匹配导致后续的运算错误

        self.room_n = room_num
        self.room_list = [
            self.ETPRoom(ki_lst[room_index], kl_lst[room_index], ko_lst[room_index], ca_lst[room_index],
                         cm_lst[room_index], ti_ini_lst[room_index], ti_ini_lst[room_index]+1)
            for room_index in range(room_num)]  # ETP模型为基础的房间初始化
        self.ti_list = ti_ini_lst
        self.tm_list = ti_ini_lst
        self.q_list = np.zeros(room_num)  # 园区各个房间的状态列表
        self.Cen_Agent = MATD3Method.MATD3(room_num, actor_lr, critic_lr, gamma, tau,
                                           save_pth, norm_variance, noise_scale, delay)  # 中央控制器的初始化
        self.Replay_Buffer = MATD3Method.ReplayBuffer(capacity)
        self.power_limit_list = [c_p * 2.5 for c_p in center_power_list]  # 功率上限
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
        for rm_i in range(self.room_n):
            self.ti_list[rm_i], self.tm_list[rm_i] = self.room_list[rm_i].cal_temp(to_last, self.q_list[rm_i])
        return self.ti_list, self.tm_list

    def area_reset(self):
        # 重置园区房间的环境
        for room in self.room_list:
            room.reset()
        self.done = False

    def env_step(self, ti_ref_list, t_o):
        # 中央控制器对园区内房间进行功率分配，并更新温度，同时负责进行强化学习数据的存储
        states_now = []
        for i in range(self.room_n):
            state = [self.room_list[i].ti / 50, ti_ref_list[i] / 50, t_o / 50]
            states_now.append(state)
        s0 = np.array(states_now)
        actions_now_net = self.Cen_Agent.take_actions(states_now)  # 动作量，包含每个房间的空调功率，归一化处理
        actions_now = []
        room_Q_list = []
        r_index = 0
        for act in actions_now_net:
            act_data = act.astype(np.float32)
            actions_now.append(act_data.tolist())
            act_Q = [act_d * self.power_limit_list[r_index] for act_d in act_data]
            room_Q_list.append(act_Q[0])
            r_index += 1
        self.q_list = room_Q_list
        temp_next, _ = self.temp_update(t_o)  # 房间温度与状态的更新
        reward_list = []
        done_list = []
        all_fit_flag = 1
        for i in range(self.room_n):
            if abs(self.room_list[i].ti - ti_ref_list[i]) > 3:
                reward_1 = - (self.room_list[i].ti - ti_ref_list[i])**2 / 360 - 0.015
                all_fit_flag = 0
            elif abs(self.room_list[i].ti - ti_ref_list[i]) > 1:
                reward_1 = -abs(self.room_list[i].ti - ti_ref_list[i]) / 120 - 0.015
                all_fit_flag = 0
            else:
                reward_1 = 0.1 - abs(self.room_list[i].ti - ti_ref_list[i]) / 100
            if abs(room_Q_list[i]) >= 0.75 * self.power_limit_list[i]:
                reward_1 -= 0.4  # actor输出层激活函数为tanh，对于接近1的值难以有效训练，所以要尽可能惩罚边界值
            if not self.done:
                done_list.append(1)
            else:
                done_list.append(0)
            reward_list.append(reward_1)
        if all_fit_flag == 1:
            reward_list = [rl + 0.2 * self.room_n for rl in reward_list]
        # 奖赏值定义，取决于温度
        self.Replay_Buffer.add(s0, actions_now, reward_list, done_list)  # 将状态量、动作量与奖赏值合并，放入智能体的经验回放池中便于学习

        return self.q_list, reward_list[0]  # 返回值是各个房间的功率列表与智能体得到的奖赏值
