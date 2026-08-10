# leo_env.py - 低轨卫星跳波束资源调度环境
# 章节映射: 第1章 (1.1 ~ 1.5)  公式 (1) ~ (24)
import numpy as np #矩阵数学 无需多言
from scipy.special import jv  # 贝塞尔函数，用于公式(3)
import random
from collections import deque   # <--- 新增导入 为了跟踪数据包进入时隙
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
        # ---------- 4. 队列与统计变量 (对应 1.4 ~ 1.5 节) ----------
        # 将原有的整型 numpy 数组替换为包含 N 个 deque 的列表
        # 每个 deque 存放该波位当前所有积压数据包的到达时隙 (int)
        self.rt_queue_timestamps = [deque() for _ in range(self.N)]  # 实时队列 ψ1 时间戳
        self.nrt_queue_timestamps = [deque() for _ in range(self.N)]  # 非实时队列 ψ2 时间戳

        # 兼容性属性：保留 realtime_queue / nrt_queue 方便直接读取当前队列长度
        # @property
        # def realtime_queue(self):
        #     return np.array([len(q) for q in self.rt_queue_timestamps], dtype=np.int32)
        #
        # @property
        # def nrt_queue(self):
        #     return np.array([len(q) for q in self.nrt_queue_timestamps], dtype=np.int32)


        self.base_demand = np.array([800, 700, 1300, 300, 980, 250,
                                     1000, 275, 80, 600, 50, 200]) #图6 估测数据
        # 2. 严格按论文 1.4 节公式计算空间离散系数 zeta (变异系数 std/mean)
        mean_demand = np.mean(self.base_demand)
        std_demand = np.std(self.base_demand)
        self.zeta = std_demand / mean_demand  # 算得约 0.7205
        #  归一化空间不均匀因子向量 (这里的归一化 指 均值为 1)
        self.spatial_factor = self.base_demand / mean_demand #图6每个波位的占比*12
        # 4. 图 7 提取的 24 小时归一化时间业务量曲线 (1:00 ~ 24:00)
        self.FIG7_TIME_PROFILE = np.array([
            0.03, 0.03, 0.03, 0.03, 0.03, 0.03,  # 1:00 - 6:00
            0.15, 0.26, 0.42, 0.60, 1.00, 0.90,  # 7:00 - 12:00 (11:00达最高峰)
            0.85, 0.78, 0.66, 0.78, 0.82, 0.68,  # 13:00 - 18:00 (17:00达次高峰)
            0.42, 0.32, 0.18, 0.10, 0.06, 0.03   ])#图7 估测数据
        self.total_slots_per_day = 8_640_000  # 一天时隙8640000个
        self.base_packet_rate = 50# 基准包到达率 (每个 10ms 时隙平均到达的包数  L：？ )

        # 实时队列 ψ_1 (时延敏感) 和非实时队列 ψ_2 (吞吐量敏感)
        # self.realtime_queue = np.zeros(self.N, dtype=np.int32)  # 积压的实时包个数
        # self.nrt_queue = np.zeros(self.N, dtype=np.int32)  # 积压的非实时包个数 L：修改为静态数组后没有必要了

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
        用以计算 self对象 由在之前的position 得出其他波位对自己的I_mn/P_m
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



                    G_theta = (10**(self.G_t/10))*((J1 / (2 * u_mn) + 36 * J3 / (u_mn ** 3)) ** 2)

                # ----- 公式(2): 干扰功率 I_mn -----
                # g_m * P_m 近似为总功率/波束数 平均分配，此处用标准功率归一化
                # 为了简化，令 g_m * P_m = 1 (相对值)，最终干扰只看空间几何 L:放你娘的屁
                # 实际实现中，由于功率分配在 step 中动态计算，这里只算几何衰减因子
                # 因此 interference[i][j] 存储的是 增益平方 * 路径损耗因子
                # 路径损耗: (λ / (4π * d_ij))^2  实际公式已有 λ^2/(4πd)^2
                path_loss =  (self.lambda_wave / (4 * np.pi * d_horizontal_ij_m)) ** 2

                # 组合: 干扰因子 (相对值，等于I_m/ P_m,后续乘以实际功率Pm ) G_m * G_theta * path_loss
                interference[i][j] =  G_theta * path_loss

        return interference  #L审核完成 最终返回二维数组 m对n的影响因子

    # ========================================================================
    # 3. 业务到达率生成 (对应 1.4 节 公式8, 图6, 图7) L：算出实时与实时业务量的均值 random模拟出实际状况
    # ========================================================================
    # def _get_traffic_rates(self, current_slot):
    #     """L重写 抄录图6 数据 手算zeta"""
    #
    #     """
    #     根据空间离散系数 ζ 和时间加权因子生成当前时隙的泊松到达率
    #
    #     返回:
    #         lambda_rt (np.array): 12个波位的实时包到达率 (包/时隙)
    #         lambda_nrt (np.array): 12个波位的非实时包到达率 (包/时隙)


    def _get_traffic_rates(self, current_slot):
            # 映射当前时隙对应图 7 的具体小时 (0 ~ 23)
        hour_idx = int((current_slot / self.total_slots_per_day) * 24) % 24
        time_factor = self.FIG7_TIME_PROFILE[hour_idx] #提取这个时段的占比 所谓的什么因子

        # 各波位的期望包到达率
        # 即当前时隙该波位所有数据包的总期望到达率
        total_expected_rate = self.spatial_factor * time_factor * self.base_packet_rate


        """区分 RT 与 NRT (1:1 分配)这里是假设服务的为1：1 因为文中没有讲 一个个尝试过去 此时暂定0.5：0.5希望我没猜错
        np.maximum(..., 0.0) 的作用是下界截断 确保大于0.0 """
        lambda_rt_expected = np.maximum(total_expected_rate * 0.5, 0.0)
        lambda_nrt_expected = np.maximum(total_expected_rate * 0.5, 0.0)

        return lambda_rt_expected, lambda_nrt_expected

    # ========================================================================
    # 4. 环境重置 (对应 算法1 步骤9)
    # ========================================================================
    def reset(self):
        # 清空时间戳队列
        self.rt_queue_timestamps = [deque() for _ in range(self.N)]
        self.nrt_queue_timestamps = [deque() for _ in range(self.N)]

        # 清空统计累加器
        self.cumulative_served = np.zeros(self.N)
        self.cumulative_demanded = np.zeros(self.N)

        # 重置时间
        self.current_slot = 0

        # 生成初始到达率
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)

        return self._get_state()

    # ========================================================================
    # 5. 状态构造 (对应 公式19~24) L用于生成矩阵
    # ========================================================================
    def _get_state(self):
        # 提取各个波位当前的队列长度
        rt_lengths = [len(q) for q in self.rt_queue_timestamps]
        nrt_lengths = [len(q) for q in self.nrt_queue_timestamps]

        packet_matrix = np.vstack([
            np.array(rt_lengths, dtype=np.float32),
            np.array(nrt_lengths, dtype=np.float32)
        ])  # shape: (2, 12)

        satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)
        satisfaction = np.clip(satisfaction, 0.0, 1.0)

        return {
            'packet_matrix': packet_matrix,
            'satisfaction': satisfaction
        }


    # ========================================================================
    # 6. 计算平均时延 (对应论文 公式 9)
    # ========================================================================

    @property
    def realtime_queue(self):
        return np.array([len(q) for q in self.rt_queue_timestamps], dtype=np.int32)

    @property
    def nrt_queue(self):
        return np.array([len(q) for q in self.nrt_queue_timestamps], dtype=np.int32)

    def _calculate_avg_delay(self, capacity_packets=None):
        """
        根据公式(9)精确计算实时数据包的平均排队时延 (秒/ms)

        参数:
            capacity_packets (dict, optional): 当前时隙各波位的服务容量(包数)
        """
        total_rt_packets = np.sum(self.realtime_queue)
        if total_rt_packets == 0:
            return 0.0

        # 方法 A: 若传入了当前时隙的实际服务容量 (更符合 Little 定律与物理传输)
        if capacity_packets is not None and sum(capacity_packets.values()) > 0:
            total_capacity = sum(capacity_packets.values())
            # 平均排队时延 = (总积压包数 / 总服务速率) * 时隙长度
            avg_delay = (total_rt_packets / total_capacity) * self.slot_duration
        else:
            # 方法 B: 理论公式(9) —— 队列积压包数 * 单时隙时长
            # 表示当前积压的实时包在平均情况下需要等待的累积时隙数
            avg_delay = np.mean(self.realtime_queue) * self.slot_duration

        return float(avg_delay)
    # ========================================================================
    # 7. 核心: 执行一步动作 (对应 算法1 步骤11~12)
    # ========================================================================
    def step(self, action):
        # ------ (0) 动作前置校验 (确保选出 K 个波位) ------
        action = list(set(action))
        if len(action) < self.K:
            remaining = [i for i in range(self.N) if i not in action]
            action += random.sample(remaining, self.K - len(action))
        action = action[:self.K]

        # ------ (1) 精确计算各波位当前的物理排队时延 & 功率分配 (公式18) ------
        current_rt_delays = np.zeros(self.N)  # 记录各波位实时包平均排队时延 (秒)

        for i in range(self.N):
            rt_q = self.rt_queue_timestamps[i]
            if len(rt_q) > 0:
                # 等待时隙数 = 当前时隙 - 入队时隙
                waiting_slots = self.current_slot - np.array(rt_q)
                # 转化为物理时间 (秒)
                current_rt_delays[i] = np.mean(waiting_slots) * self.slot_duration
            else:
                current_rt_delays[i] = 0.0

        # 根据公式 (18) 计算功率权重
        weights = {}
        for i in action:
            total_packets = len(self.rt_queue_timestamps[i]) + len(self.nrt_queue_timestamps[i])
            # delay_weight 严格采用上面算出的排队时延 (单位: 秒)
            delay_weight = current_rt_delays[i]

            # W_i = (总包数 + 1) * (时延 + 防零平滑项)
            weights[i] = (total_packets + 1) * (delay_weight + 1e-5)

        total_weight = sum(weights.values())
        allocated_power = {}
        for i in action:
            p_i = (weights[i] / total_weight) * self.total_power
            allocated_power[i] = min(p_i, self.max_beam_power)

        # ------ (2) 干扰计算与信道容量 (公式2~7) ------
        capacity_packets = {}
        for i in action:
            interference_sum = 0.0
            for j in action:
                if i != j:
                    interference_sum += self.interference_matrix[i][j] * allocated_power[j]

            noise_power = Bo * T_noise * self.bandwidth

            # 考虑到卫星到波位 i 的直达路径损耗 d_i
            xi_m, yi_m = self.spot_positions[i] * 1000.0  # km 转换为 m
            d_i_m = np.sqrt(xi_m**2 + yi_m**2 + self.h**2)  # 全米单位计算
            path_loss_i = (self.lambda_wave / (4 * np.pi * d_i_m)) ** 2

            # 有用信号功率 P_i * G_t * G_r * PathLoss
            signal_power = allocated_power[i] * (10 ** (self.G_t / 10)) * (10 ** (self.G_r / 10)) * path_loss_i


            sinr = signal_power / (interference_sum + noise_power)

            capacity_bps = self.bandwidth * np.log2(1 + sinr)
            max_packets = (capacity_bps * self.slot_duration) / self.packet_size
            capacity_packets[i] = int(np.floor(max_packets))

        # ------ (3) 队列更新: 先入队 (泊松到达 & 记录到达时隙) ------
        self.lambda_realtime, self.lambda_nrt = self._get_traffic_rates(self.current_slot)

        for i in range(self.N):
            arrive_rt = np.random.poisson(self.lambda_realtime[i])
            arrive_nrt = np.random.poisson(self.lambda_nrt[i])

            # 将新到达的数据包其到达时隙 (self.current_slot) 压入队列
            for _ in range(arrive_rt):
                self.rt_queue_timestamps[i].append(self.current_slot)
            for _ in range(arrive_nrt):
                self.nrt_queue_timestamps[i].append(self.current_slot)

            # 更新累计需求 (公式11 分母)
            self.cumulative_demanded[i] += (arrive_rt + arrive_nrt)

        # ------ (4) 队列更新: 后出队 (先实时后非实时, FIFO 弹出最老的包) ------
        served_realtime_total = 0
        served_nrt_total = 0

        for i in action:
            cap = capacity_packets[i]

            # 1. 服务实时包 (FIFO: 从左侧弹出最先进入的包)
            served_rt = min(len(self.rt_queue_timestamps[i]), cap)
            for _ in range(served_rt):
                self.rt_queue_timestamps[i].popleft()
            cap -= served_rt
            served_realtime_total += served_rt

            # 2. 剩余容量服务非实时包
            served_nrt = min(len(self.nrt_queue_timestamps[i]), cap)
            for _ in range(served_nrt):
                self.nrt_queue_timestamps[i].popleft()
            served_nrt_total += served_nrt

            # 更新累计服务 (公式24 分子)
            self.cumulative_served[i] += (served_rt + served_nrt)

        # ------ (5) 丢包处理 (超时丢弃最老包, 公式16) ------
        # 超时阈值对应的时隙个数 T_th / T_slot
        max_slots_threshold = int(self.delay_threshold / self.slot_duration)  # 0.4s / 0.01s = 40 时隙

        for i in range(self.N):
            # 丢弃实时队列里在队列中滞留超过 40 个时隙的数据包
            while len(self.rt_queue_timestamps[i]) > 0:
                oldest_packet_slot = self.rt_queue_timestamps[i][0]
                if (self.current_slot - oldest_packet_slot) > max_slots_threshold:
                    self.rt_queue_timestamps[i].popleft()  # 丢弃超时数据包
                else:
                    break  # 队头最老的包都没超时，后面的包肯定也没超时

        # ------ (6) 计算单目标奖励 (公式9 / 公式10 / 公式11) ------
        if self.objective == 'throughput':
            reward = float(served_nrt_total)


        elif self.objective == 'delay': #为了防止较多积压时，reward 为负数 加入防爆项与平滑截断

            total_rt_packets = sum([len(q) for q in self.rt_queue_timestamps])

            if total_rt_packets > 0:

                system_avg_delay = np.sum(

                    [current_rt_delays[i] * len(self.rt_queue_timestamps[i]) for i in range(self.N)]

                ) / total_rt_packets

            else:

                system_avg_delay = 0.0

            # 规范化到 [-1, 0] 区间，避免极值爆炸

            normalized_delay = min(system_avg_delay / self.delay_threshold, 2.0)

            reward = -float(normalized_delay)

        elif self.objective == 'satisfaction':
            satisfaction = self.cumulative_served / (self.cumulative_demanded + 1e-8)
            reward = float(np.mean(satisfaction))

        # ------ (7) 时隙递增与状态生成 ------
        self.current_slot += 1
        next_state = self._get_state()
        done = False

        # ------ (8) 返回 info ------
        info = {
            'avg_delay': np.mean(current_rt_delays),  # 各波位平均排队时延 (s)
            'throughput': served_nrt_total,
            'satisfaction': np.mean(self.cumulative_served / (self.cumulative_demanded + 1e-8)),
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