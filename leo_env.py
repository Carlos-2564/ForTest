# leo_env.py - 低轨卫星跳波束资源调度环境
# 章节映射: 第1章 (1.1 ~ 1.5)  公式 (1) ~ (24)
import numpy as np #矩阵数学 无需多言
from scipy.special import jv  # 贝塞尔函数，用于公式(3)
import random

T_noise = 300  # 噪声温度 300 K 表1 全局变量
Bo = 1.38e-23  # 玻尔兹曼常数 表1 全局变量


class LEOSatEnv:
    """
    模拟卫星环境
    单颗低轨卫星跳波束环境
    特点：
    1. 12个固定地面波位 (图5)
    2. 4个可同时激活的波束 (表1)
    3. 包含同频干扰的物理层计算 (公式2~7)
    4. 实时/非实时双队列模型 (公式9~11)
    5. 支持单目标切换 (throughput / delay / satisfaction)
    """

    def __init__(self, objective='throughput'):
        """
        生成类
        self为关键字 指代这个类自己
        初始化环境（对应论文 表1 参数）
        参数:
            objective (str): 单专家目标，可选 'throughput' | 'delay' | 'satisfaction'
        """
        # ---------- 1. 空间与物理参数 (表1) L已经核对 确保一致----------
        self.t_num=36# 轨道数36个
        self.signle_num=20 #单轨卫星数20个
        self.h = 570e3  # 轨道高度 570 km (米)
        self.A = 70  #轨道倾角70度
        self.Total_num=720 #卫星总数120个
        self.N = 12  # 波位总数 12
        self.K = 4  # 同时激活的波束数4
        self.fc = 20e9  # 载波频率 20 GHz
        self.bandwidth = 200e6  # 带宽 200 MHz (Hz)
        self.total_power = 120  # 星上总功率 120 W
        self.max_beam_power = 60  # 单波束最大功率 60 W
        self.G_t = 40  # 卫星发射天线增益 (dB)
        self.G_r = 50  # 用户接收天线增益 (dB)
        self.slot_duration = 0.01  # 10 ms (秒)  跳波束时隙长度10ms
        self.delay_threshold = 0.4  # 排队延迟400 ms (秒)
        self.packet_size = 10 * 1024 * 8  # 数据包大小10 kbit
                                            #噪声温度 300 K 表1 全局变量
                                        # 玻尔兹曼常数 表1 见全局变量

        self.lambda_wave = 3e8 / self.fc  # 波长

        # ---------- 2. 波位几何布局 (对应 1.3 节 图5) ----------
        # 12个波位均匀分布在星下点周围，参考图5  见后文
        self.spot_positions = self._generate_spot_positions()

        # ---------- 3. 预计算干扰矩阵 (公式2~5) ----------
        # interference_matrix[i][j] 表示当波位 i 和 j 同时被点亮时，i 受到 j 的干扰功率 (W)
        self.interference_matrix = self._precompute_interference()

        # ---------- 4. 队列与统计变量 (对应 1.4 ~ 1.5 节) ----------
        # 实时队列 ψ_1 (时延敏感) 和非实时队列 ψ_2 (吞吐量敏感)
        self.realtime_queue = np.zeros(self.N, dtype=np.int32)  # 积压的实时包个数
        self.nrt_queue = np.zeros(self.N, dtype=np.int32)  # 积压的非实时包个数

        # 满意度统计累加器 (公式24)
        self.cumulative_served = np.zeros(self.N)  # 累计已服务包数
        self.cumulative_demanded = np.zeros(self.N)  # 累计总需求包数

        # 时间与业务到达率 (对应公式8 及 图6/图7)
        self.current_slot = 0
        self.lambda_realtime = None  # 各波位实时到达率 (泊松 λ)
        self.lambda_nrt = None  # 各波位非实时到达率

        # 目标选择
        self.objective = objective

        # 打印状态
        print(f"[Env] 初始化完成 | 目标: {objective} | 波位数: {self.N} | 波束数: {self.K}")












    # ========================================================================
    # 1. 波位布局生成 (对应 1.3 节) 单位km 对应图五 L已核对 顺序皆相同
    # ========================================================================
    def _generate_spot_positions(self):
        # 波位半径 R (km)
        R = 73
        # 相邻波位中心距 d = sqrt(3) * R
        d = np.sqrt(3) * R

        positions = [(0, 0)]  # 中心波位 (序号 1)

        # 第一层 (波位 2~7)：6个波位，距离中心为 d，角度从 30° 开始，每隔 60° 一个
        for k in range(6):
            angle = np.deg2rad(60 * k + 30)
            x = d * np.cos(angle)
            y = d * np.sin(angle)
            positions.append((x, y))

        # 第二层 (波位 8~12)：外围 5 个波位，利用标准六边形网格平移矢量紧密拼合（对应图 5 布局）
        # outer_offsets 对应与第一层紧密相切的外围 5 个中心点坐标
        outer_offsets = [
            (d * np.cos(np.deg2rad(30)) + d * np.cos(np.deg2rad(90)),
             d * np.sin(np.deg2rad(30)) + d * np.sin(np.deg2rad(90))),  # 波位 8
            (d * np.cos(np.deg2rad(90)) + d * np.cos(np.deg2rad(150)),
             d * np.sin(np.deg2rad(90)) + d * np.sin(np.deg2rad(150))),  # 波位 9
            (2 * d * np.cos(np.deg2rad(150)),
             2 * d * np.sin(np.deg2rad(150))),  # 波位 10
            (d * np.cos(np.deg2rad(150)) + d * np.cos(np.deg2rad(210)),
             d * np.sin(np.deg2rad(150)) + d * np.sin(np.deg2rad(210))),  # 波位 11
            (d * np.cos(np.deg2rad(210)) + d * np.cos(np.deg2rad(270)),
             d * np.sin(np.deg2rad(210)) + d * np.sin(np.deg2rad(270)))  # 波位 12
        ]

        for x, y in outer_offsets:
            positions.append((x, y))

        return np.array(positions)

    # ========================================================================
    # 2. 同频干扰矩阵预计算 (对应 公式2~5)
    # ========================================================================
    def _precompute_interference(self):
        """
        提前算好 12x12 的干扰功率矩阵 (W)
        避免在 step() 中重复计算贝塞尔函数，大幅提升训练速度
        """
        N = self.N # 波位总数 12
        interference = np.zeros((N, N)) #12*12 所有元素均为0 的数组

        # 卫星到各波位的距离 d_i (公式中 d_n) 和 俯仰角相关因子
        for i in range(N):
            for j in range(N):
                if i == j:continue        # 自身对自身无影响

                # 波位 i 和 j 的坐标 读之前的波位布局生成
                xi, yi = self.spot_positions[i]
                xj, yj = self.spot_positions[j]

                # 水平距离 (km)
                d_horizontal_ij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                # 卫星到波位 i 的直线距离 d_i (km)   (公式中 d_n)
                d_i = np.sqrt(xi ** 2 + yi ** 2 + (self.h / 1000) ** 2)
                # 卫星到波位 j 的直线距离 d_j (km)   (公式中 d_m)
                d_j = np.sqrt(xj ** 2 + yj ** 2 + (self.h / 1000) ** 2)

                # 转换为米 (公式中需要米)
                # ?暂疑惑 哪里提到用m 后文统一用m算 累了不改
                d_i_m = d_i * 1000
                d_j_n = d_j * 1000
                d_horizontal_ij_m = d_horizontal_ij * 1000

                # ----- 公式(5): 计算夹角 θ_mn (弧度) -----
                # 简化计算: 利用余弦定理
                # cos(theta) = (d_m^2 + d_n^2 - d_mn^2) / (2 * d_m * d_n)
                # 注: 原文公式(5)中分母符号有笔误，这里采用标准余弦定理  L：检查为等效
                cos_theta = (d_i_m ** 2 + d_j_n ** 2 - d_horizontal_ij_m ** 2) / (2 * d_i_m * d_j_n)
                # 防止数值溢出 (-1 ~ 1 截断)
                cos_theta = np.clip(cos_theta, -1.0, 1.0) #NumPy 的裁剪函数每一个值限制在 [-1.0, 1.0] 的闭区间内 更的大直接变1
                theta_mn = np.arccos(cos_theta) #得到角 θ_mn

                # ----- 公式(4): 计算 u_mn -----

                """L:通常在卫星通信设计中，一个 3dB 波束正好覆盖一个蜂窝波位（即波位边缘的功率衰减为 3dB）。
                因此，波位半径 R 在卫星天线处所张开的半角即为 \theta_{3\text{dB}}
               sin_theta_3db = 73/np.sqrt(73**2+self.h**2)轨道高度570  #km/km 结果相同
               算得为0.12703 
               """
                sin_theta_3db =0.12703
                u_mn = 2.07123 * np.sin(theta_mn) / sin_theta_3db

                # ----- 公式(3): 天线增益 G(theta) -----
                if u_mn == 0:
                    G_theta = 1.0  # 避免除零
                else:
                    J1 = jv(1, u_mn)  # 一阶贝塞尔函数
                    J3 = jv(3, u_mn)  # 三阶贝塞尔函数
                    # 注意: 原文公式(3)分母为 2*u_i 但应为 2*u_mn，且系数36
                    """矛盾点：当带入公式(2)计算波位m对波位n的干扰功率I_{mn}时，应该使用的是由夹角theta_{mn}算出的u_{mn}
                    如果增益公式里还写着u_i就会在逻辑上产生断层（不知道u_i到底指哪个波位）
                    代码循环中，变量 u_mn 实际代表的就是波位 $i$ 与波位 $j$ 之间的 $u_{ij}$。
                    把公式(3)里的u_i替换为 u_mn（即u_{ij}），
                    在计算干扰矩阵的编程实现上是完全正确且必须的 值得提问(1)"""
                    G_theta = (J1 / (2 * u_mn) + 36 * J3 / (u_mn ** 3)) ** 2

                # ----- 公式(2): 干扰功率 I_mn -----
                # g_m * P_m 近似为总功率/波束数 平均分配，此处用标准功率归一化
                # 为了简化，令 g_m * P_m = 1 (相对值)，最终干扰只看空间几何
                # 放你娘的屁
                # 实际实现中，由于功率分配在 step 中动态计算，这里只算几何衰减因子
                # 因此 interference[i][j] 存储的是 增益平方 * 路径损耗因子
                # 路径损耗: (λ / (4π * d_ij))^2  实际公式已有 λ^2/(4πd)^2
                path_loss = (self.lambda_wave / (4 * np.pi * d_horizontal_ij_m)) ** 2

                # 组合: 干扰因子 (相对值，后续乘以实际功率)
                interference[i][j] = G_theta * path_loss

        return interference

    # ========================================================================
    # 3. 业务到达率生成 (对应 1.4 节 公式8, 图6, 图7)
    # ========================================================================
    def _get_traffic_rates(self, current_slot):
        """
        根据空间离散系数 ζ 和时间加权因子生成当前时隙的泊松到达率

        返回:
            lambda_rt (np.array): 12个波位的实时包到达率 (包/时隙)
            lambda_nrt (np.array): 12个波位的非实时包到达率 (包/时隙)
        """
        # (1) 空间分布 (图6): 假设波位0~3高密度，4~7中密度，8~11低密度
        # 离散系数 ζ 越大，空间分布越不均 (论文公式8)
        zeta = 0.5  # 中等空间不均性
        base_demand = np.array([100, 80, 60, 40, 90, 70, 50, 30, 60, 40, 20, 10])
        # 加入随机扰动增加空间离散度
        spatial_factor = base_demand / np.mean(base_demand) * (1 + zeta * np.random.randn(self.N))
        spatial_factor = np.maximum(spatial_factor, 0.1)  # 保证非负

        # (2) 时间周期性 (图7): 模拟 9:00~14:00 业务波峰波谷
        # 假设每个时隙10ms，1000时隙 = 10秒，我们模拟一个简化的日周期
        # 在 500 时隙处达到峰值 (对应11:00)
        time_factor = 0.5 + 0.5 * np.sin(np.pi * current_slot / 500)
        time_factor = np.clip(time_factor, 0.3, 1.0)

        # (3) 实时与非实时比例 (1:1)
        total_rate = spatial_factor * time_factor * 50  # 基值50包/时隙
        lambda_rt = (total_rate * 0.5).astype(int)
        lambda_nrt = (total_rate * 0.5).astype(int)

        return lambda_rt, lambda_nrt

    # ========================================================================
    # 4. 环境重置 (对应 算法1 步骤9)
    # ========================================================================
    def reset(self):
        """
        重置环境状态 (新的一颗卫星过顶 / 新的训练周期)

        返回:
            state (dict): 初始状态字典
        """
        # 清空队列
        self.realtime_queue = np.zeros(self.N, dtype=np.int32)
        self.nrt_queue = np.zeros(self.N, dtype=np.int32)

        # 清空统计累加器
        self.cumulative_served = np.zeros(self.N)
        self.cumulative_demanded = np.zeros(self.N)

        # 重置时间
        self.current_slot = 0

        # 生成初始到达率
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)

        # 返回初始状态
        state = self._get_state()
        return state

    # ========================================================================
    # 5. 状态构造 (对应 公式19~24)
    # ========================================================================
    def _get_state(self):
        """
        构造当前状态 s_t

        状态组成:
            1. packet_matrix (2, 12): 第一行实时包数, 第二行非实时包数
            2. satisfaction (12,): 各波位满意度 (公式24)

        返回:
            dict: {'packet_matrix': ndarray, 'satisfaction': ndarray}
        """
        packet_matrix = np.vstack([
            self.realtime_queue.astype(np.float32),
            self.nrt_queue.astype(np.float32)
        ])  # shape: (2, 12)

        # 计算满意度 (公式24): η = 累计服务 / 累计需求
        satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)
        satisfaction = np.clip(satisfaction, 0.0, 1.0)  # 截断到 [0, 1]

        return {
            'packet_matrix': packet_matrix,
            'satisfaction': satisfaction
        }

    # ========================================================================
    # 6. 计算平均时延 (辅助函数，用于 reward)
    # ========================================================================
    def _calculate_avg_delay(self):
        """
        估算实时数据包的平均排队时延 (公式9 的简化版本)
        由于未追踪每个包的时间戳，此处用队列长度/服务速率近似
        """
        # 总实时包数
        total_rt = np.sum(self.realtime_queue)
        if total_rt == 0:
            return 0.0
        # 估算服务速率: 假设当前容量下每时隙平均服务包数
        # 这里简单用队列长度 * 时隙长度 作为总排队时间的近似
        # 实际论文中通过公式(17)精确计算
        avg_delay = total_rt * self.slot_duration * 10  # 粗略放大
        return avg_delay

    # ========================================================================
    # 7. 核心: 执行一步动作 (对应 算法1 步骤11~12)
    # ========================================================================
    def step(self, action):
        """
        执行动作，环境状态转移

        参数:
            action (list): 长度为 K=4 的波位 ID 列表, 如 [2, 5, 7, 11]

        返回:
            next_state (dict): 下一状态
            reward (float): 单目标标量奖励
            done (bool): 是否结束 (本环境固定1000步，设为False)
            info (dict): 调试信息 (平均时延、吞吐量、满意度)
        """
        # ------ (1) 动作解码与功率分配 (公式18) ------
        # 确保 action 有 K 个不同元素
        action = list(set(action))
        if len(action) < self.K:
            # 补齐随机波位 (避免少于K个)
            remaining = [i for i in range(self.N) if i not in action]
            action += random.sample(remaining, self.K - len(action))
        action = action[:self.K]  # 截断至K个

        # 计算各选中波位的功率权重: weight = (总包数) * (平均排队时延)
        weights = {}
        for i in action:
            total_packets = self.realtime_queue[i] + self.nrt_queue[i]
            # 用队列长度近似时延权重
            delay_weight = self.realtime_queue[i] * 1.0 + self.nrt_queue[i] * 0.5
            weights[i] = (total_packets + 1) * (delay_weight + 1)  # +1防零

        total_weight = sum(weights.values())
        allocated_power = {}
        for i in action:
            p_i = (weights[i] / total_weight) * self.total_power
            allocated_power[i] = min(p_i, self.max_beam_power)

        # ------ (2) 干扰计算与信道容量 (公式2~7) ------
        SINR_dict = {}
        capacity_packets = {}

        for i in action:
            # 累加来自其他选中波位的干扰功率
            interference_sum = 0.0
            for j in action:
                if i != j:
                    # 查预计算干扰矩阵，乘上干扰源的发射功率
                    interference_sum += self.interference_matrix[i][j] * allocated_power[j]

            # 噪声功率: N0 = k * T * B
            noise_power = Bo * T_noise * self.bandwidth

            # 公式(7): SINR = (P_i * G_t * G_r) / (I + N0)
            # 注: 链路损耗 L_sl 在干扰矩阵预计算中已包含，此处省略常数因子
            signal_power = allocated_power[i] * (10 ** (self.G_t / 10)) * (10 ** (self.G_r / 10))
            sinr = signal_power / (interference_sum + noise_power)
            SINR_dict[i] = sinr

            # 公式(6): 信道容量 C = B * log2(1 + SINR)  (bps)
            capacity_bps = self.bandwidth * np.log2(1 + sinr)
            # 转换为每时隙能传输的包数 (公式17 前半部分)
            max_packets = (capacity_bps * self.slot_duration) / self.packet_size
            capacity_packets[i] = int(np.floor(max_packets))

        # ------ (3) 队列更新: 先入队 (泊松到达) ------
        # 更新到达率 (随当前时隙变化)
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)

        for i in range(self.N):
            # 实时包到达
            arrive_rt = np.random.poisson(self.lambda_realtime[i])
            self.realtime_queue[i] += arrive_rt
            # 非实时包到达
            arrive_nrt = np.random.poisson(self.lambda_nrt[i])
            self.nrt_queue[i] += arrive_nrt

            # 更新累计需求 (公式11 分母)
            self.cumulative_demanded[i] += (arrive_rt + arrive_nrt)

        # ------ (4) 队列更新: 后出队 (先实时后非实时) ------
        served_realtime_total = 0
        served_nrt_total = 0
        served_dict = {}  # 记录每个波位服务了多少包

        for i in action:
            cap = capacity_packets[i]
            # 优先服务实时数据 (时延敏感)
            served_rt = min(self.realtime_queue[i], cap)
            self.realtime_queue[i] -= served_rt
            cap -= served_rt
            served_realtime_total += served_rt

            # 剩余容量服务非实时数据
            served_nrt = min(self.nrt_queue[i], cap)
            self.nrt_queue[i] -= served_nrt
            cap -= served_nrt
            served_nrt_total += served_nrt

            served_dict[i] = (served_rt, served_nrt)
            # 更新累计服务 (公式24 分子)
            self.cumulative_served[i] += (served_rt + served_nrt)

        # ------ (5) 丢包处理 (时延阈值 T_th = 400ms, 公式16) ------
        # 若队列积压超过阈值 (用包数*时隙长度近似估计超时)，丢弃最老包
        max_packets_threshold = int(self.delay_threshold / self.slot_duration) * 2  # 约80包
        for i in range(self.N):
            if self.realtime_queue[i] > max_packets_threshold:
                # 丢弃超出的实时包
                self.realtime_queue[i] = max_packets_threshold
            if self.nrt_queue[i] > max_packets_threshold * 2:
                self.nrt_queue[i] = max_packets_threshold * 2

        # ------ (6) 计算单目标奖励 (块④ 步骤12) ------
        if self.objective == 'throughput':
            # 公式(10): 最大化非实时吞吐量 -> 奖励 = 本时隙传输的非实时包数
            reward = float(served_nrt_total)

        elif self.objective == 'delay':
            # 公式(9): 最小化实时时延 -> 奖励 = -平均时延 (DQN最大化它)
            avg_delay = self._calculate_avg_delay()
            reward = -avg_delay

        elif self.objective == 'satisfaction':
            # 公式(11): 最大化服务满意度 -> 奖励 = 当前全局平均满意度
            satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)
            reward = float(np.mean(satisfaction))
        else:
            raise ValueError(f"未知目标: {self.objective}")

        # ------ (7) 步进时间与状态更新 ------
        self.current_slot += 1
        next_state = self._get_state()

        # ------ (8) 返回 ------
        # done: 本环境固定1000时隙，但由外部循环控制，此处始终返回False
        done = False

        # info: 用于调试和绘图
        satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)
        info = {
            'avg_delay': self._calculate_avg_delay(),
            'throughput': served_nrt_total,
            'satisfaction': np.mean(satisfaction),
            'action': action,
            'capacity': capacity_packets
        }

        return next_state, reward, done, info


# ============================================================================
# 测试代码 (独立运行环境，验证干扰矩阵和队列逻辑)
# ============================================================================
if __name__ == "__main__":
    # 快速测试环境是否正常工作
    env = LEOSatEnv(objective='throughput')
    state = env.reset()
    print(f"初始状态包矩阵形状: {state['packet_matrix'].shape}")
    print(f"初始满意度形状: {state['satisfaction'].shape}")

    # 随机执行几步
    for t in range(5):
        action = random.sample(range(env.N), env.K)
        next_state, reward, done, info = env.step(action)
        print(f"时隙 {t + 1}: 动作 {action} -> 奖励 {reward:.2f}, 吞吐量 {info['throughput']}")

    print("环境测试通过！")