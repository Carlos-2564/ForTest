import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from leo_env import LEOSatEnv

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 神经网络结构定义 (对应图 13 与 图 14)
# ==========================================

class ConvQNetwork(nn.Module):
    """
    对应图 13：用于 Latency 和 Throughput 的 Q 网络
    输入: state['packet_matrix'] 形状 (2, 12) -> 扩展为 (1, 2, 12) 卷积
    """

    def __init__(self, action_dim=12):
        super(ConvQNetwork, self).__init__()
        # Conv1: input channels=1, output channels=8
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 2), padding=(0, 1))
        # Conv2: input channels=8, output channels=16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

        # 💡 动态自动计算展平后的维度 (Flatten Dimension)
        dummy_input = torch.zeros(1, 1, 2, 12)
        conv_out = self.relu(self.conv1(dummy_input))
        conv_out = self.relu(self.conv2(conv_out))
        flatten_dim = conv_out.view(1, -1).size(1)  # 这里会自动算出来是 416

        # 全连接层
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        # x shape: (batch_size, 2, 12) -> (batch_size, 1, 2, 12)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

class FCQNetwork(nn.Module):
    """
    对应图 14：用于 Satisfaction 的 Q 网络
    输入: state['satisfaction'] 形状 (12,)
    """

    def __init__(self, input_dim=12, action_dim=12):
        super(FCQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# ==========================================
# 2. 单目标 DQNAgent 定义
# ==========================================

class DQNAgent:
    def __init__(self, net_type='conv', state_key='packet_matrix', lr=1e-5, gamma=0.9, buffer_capacity=3000,
                 batch_size=8):
        self.state_key = state_key
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_capacity)

        if net_type == 'conv':
            self.eval_net = ConvQNetwork().to(device)
            self.target_net = ConvQNetwork().to(device)
        else:
            self.eval_net = FCQNetwork().to(device)
            self.target_net = FCQNetwork().to(device)

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_q_values(self, state):
        s = torch.tensor(state[self.state_key], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.eval_net(s)
        return q_values.cpu().numpy()[0]

    def store_transition(self, s, a, r, s_next):
        self.buffer.append((s[self.state_key], a, r, s_next[self.state_key]))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s_batch = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).to(device)
        a_batch = [b[1] for b in batch]
        r_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(device)
        s_next_batch = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).to(device)

        # 当前 Q 值估算 (采用被选择波位 Q 值的平均或和作为估计)
        q_eval_all = self.eval_net(s_batch)
        q_eval = torch.stack([q_eval_all[i, a_batch[i]].sum() for i in range(self.batch_size)]).unsqueeze(1)

        # 目标 Q 值计算
        with torch.no_grad():
            q_next_all = self.target_net(s_next_batch)
            q_next_max = torch.stack([q_next_all[i].topk(4).values.sum() for i in range(self.batch_size)]).unsqueeze(1)
            q_target = r_batch + self.gamma * q_next_max

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


# ==========================================
# 3. 主训练循环 (对应 算法1)
# ==========================================

def train():
    # 论文表 2 训练参数
    LOOPS = 450
    LOOPS_TIME_SLOT = 1000
    G_TARGET_UPDATE = 100

    epsilon_start = 0.5
    epsilon_end = 0.01
    epsilon_decay = (epsilon_start - epsilon_end) / LOOPS

    env_latency = LEOSatEnv(objective='delay')
    env_throughput = LEOSatEnv(objective='throughput')
    env_satisfaction = LEOSatEnv(objective='satisfaction')

    # 初始化 3 个专家智能体
    agent_lat = DQNAgent(net_type='conv', state_key='packet_matrix')
    agent_thp = DQNAgent(net_type='conv', state_key='packet_matrix')
    agent_sat = DQNAgent(net_type='fc', state_key='satisfaction')

    # 记录训练数据用于绘制图 15
    history_delay = []
    history_throughput = []
    history_satisfaction = []

    print(">>> 开始 MA-DRL 训练...")

    epsilon = epsilon_start
    for loop in range(1, LOOPS + 1):
        s_lat = env_latency.reset()
        s_thp = env_throughput.reset()
        s_sat = env_satisfaction.reset()

        ep_delay = []
        ep_throughput = []
        ep_satisfaction = []

        for t in range(1, LOOPS_TIME_SLOT + 1):
            # 1. 贪婪策略或多目标动作选择
            if random.random() < epsilon:
                action = random.sample(range(12), 4)
            else:
                # 提取三个专家的 Q 值
                q1 = agent_lat.select_q_values(s_lat)
                q2 = agent_thp.select_q_values(s_thp)
                q3 = agent_sat.select_q_values(s_sat)

                # 归一化处理 (L2 范数)
                q1_norm = - (q1 / (np.linalg.norm(q1) + 1e-8))  # 时延取负
                q2_norm = q2 / (np.linalg.norm(q2) + 1e-8)
                q3_norm = q3 / (np.linalg.norm(q3) + 1e-8)

                # 按照式 (24)/(25) 线性标量化组合
                q_combined = (1 / 3) * q1_norm + (1 / 3) * q2_norm + (1 / 3) * q3_norm
                # 选择 Q 值最大的前 K=4 个波位
                action = np.argsort(q_combined)[-4:].tolist()

            # 2. 各自环境步进与数据收集
            s_lat_next, r1, _, info_lat = env_latency.step(action)
            s_thp_next, r2, _, info_thp = env_throughput.step(action)
            s_sat_next, r3, _, info_sat = env_satisfaction.step(action)

            # 3. 经验池存储
            agent_lat.store_transition(s_lat, action, r1, s_lat_next)
            agent_thp.store_transition(s_thp, action, r2, s_thp_next)
            agent_sat.store_transition(s_sat, action, r3, s_sat_next)

            # 4. 状态更新
            s_lat, s_thp, s_sat = s_lat_next, s_thp_next, s_sat_next

            # 5. 模型更新 (每 4 步更新一次)
            if t % 4 == 0:
                agent_lat.update()
                agent_thp.update()
                agent_sat.update()

            # 6. 目标网络更新
            if t % G_TARGET_UPDATE == 0:
                agent_lat.update_target_network()
                agent_thp.update_target_network()
                agent_sat.update_target_network()

            # 记录指标
            ep_delay.append(info_lat['avg_delay'])
            ep_throughput.append(info_thp['throughput'])
            ep_satisfaction.append(info_sat['satisfaction'])

        epsilon = max(epsilon_end, epsilon - epsilon_decay)

        avg_d = np.mean(ep_delay)
        avg_t = np.sum(ep_throughput) / 1000.0  # Mbits 级转换
        avg_s = np.mean(ep_satisfaction)

        history_delay.append(avg_d)
        history_throughput.append(avg_t)
        history_satisfaction.append(avg_s)

        if loop % 10 == 0 or loop == 1:
            print(
                f"Loop {loop}/{LOOPS} | Avg Delay: {avg_d:.2f} ms | Throughput: {avg_t:.2f} Mbits | Satisfaction: {avg_s:.4f}")

    # 保存模型权重及训练日志
    os.makedirs("./models", exist_ok=True)
    torch.save(agent_lat.eval_net.state_dict(), "./models/agent_lat.pth")
    torch.save(agent_thp.eval_net.state_dict(), "./models/agent_thp.pth")
    torch.save(agent_sat.eval_net.state_dict(), "./models/agent_sat.pth")
    np.savez("./models/train_history.npz", delay=history_delay, throughput=history_throughput,
             satisfaction=history_satisfaction)
    print(">>> 模型和训练历史已成功保存至 ./models/ 目录！")


if __name__ == "__main__":
    train()