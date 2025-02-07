import gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

#######################################
# 1. 自定义简化版 Chrome 恐龙游戏环境
#######################################
class ChromeDinoEnv(gym.Env):
    """
    该环境模拟一个简化版的 Chrome 恐龙游戏：
    - 恐龙始终在 x=0 位置
    - 障碍物从右侧以固定速度向左移动
    - 恐龙可以选择跳跃（action=1）或不动（action=0）
    - 恐龙跳跃时有一个简单的抛物线运动
    - 当障碍物与恐龙重叠且恐龙高度不足以躲避障碍时，判定为碰撞
    """
    def __init__(self):
        super(ChromeDinoEnv, self).__init__()
        # 定义观测空间：包含 [障碍物距离, 障碍物高度, 恐龙高度, 恐龙垂直速度]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, -20], dtype=np.float32),
            high=np.array([100, 20, 15, 20], dtype=np.float32),
            dtype=np.float32
        )
        # 定义动作空间：0 - 无操作，1 - 跳跃
        self.action_space = gym.spaces.Discrete(2)
        
        # 模拟参数
        self.obstacle_speed = 1.0      # 障碍物移动速度
        self.gravity = 1.0             # 重力
        self.jump_velocity = 10.0      # 跳跃初速度
        self.dino_x = 0.0              # 恐龙横坐标（固定为0）
        self.max_steps = 1000          # 每个episode最大步数
        self.reset()

    def reset(self):
        # 重置恐龙状态（在地面）
        self.dino_y = 0.0
        self.dino_v = 0.0
        
        # 生成一个障碍物，障碍物的 x 坐标（距离）和高度随机
        self.obstacle_x = random.uniform(50, 70)
        self.obstacle_height = random.choice([5.0, 6.0, 7.0, 8.0])
        self.obstacle_width = 2.0
        
        self.current_step = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        # 观测状态为：障碍物距离、障碍物高度、恐龙高度、恐龙垂直速度
        distance = self.obstacle_x  # 恐龙固定在0位置，障碍物距离即为其 x 坐标
        return np.array([distance, self.obstacle_height, self.dino_y, self.dino_v], dtype=np.float32)
    
    def step(self, action):
        reward = 1.0  # 每一步奖励1
        done = False
        info = {}
        
        # 处理动作
        if action == 1:  # 执行跳跃
            if self.dino_y == 0.0:  # 只有在地面时才能跳跃
                self.dino_v = self.jump_velocity
        
        # 更新恐龙状态（简单的跳跃物理）
        if self.dino_y > 0 or self.dino_v != 0:
            self.dino_y += self.dino_v
            self.dino_v -= self.gravity
            if self.dino_y < 0:
                self.dino_y = 0.0
                self.dino_v = 0.0
        
        # 更新障碍物位置
        self.obstacle_x -= self.obstacle_speed
        if self.obstacle_x < -self.obstacle_width:
            # 障碍物走出屏幕后重新生成
            self.obstacle_x = random.uniform(50, 70)
            self.obstacle_height = random.choice([5.0, 6.0, 7.0, 8.0])
        
        # 碰撞检测：若障碍物横跨 x=0 且恐龙高度不足，则判定碰撞
        if 0 >= self.obstacle_x and 0 <= self.obstacle_x + self.obstacle_width:
            if self.dino_y < self.obstacle_height:
                reward = -100.0  # 惩罚
                done = True
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        return obs, reward, done, info
    
    def render(self, mode='human'):
        # 简单的文本输出，用于观察环境状态
        print(f"Step: {self.current_step}, Dino y: {self.dino_y:.2f}, v: {self.dino_v:.2f}, Obstacle x: {self.obstacle_x:.2f}, Height: {self.obstacle_height}")
    
    def close(self):
        pass

#######################################
# 2. 定义 DQN 网络及 Replay Buffer
#######################################
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

#######################################
# 3. 训练 DQN 智能体
#######################################
def train():
    env = ChromeDinoEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化策略网络和目标网络
    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(10000)
    
    # 超参数设置
    num_episodes = 500         # 训练回合数
    batch_size = 64            # 每次采样 batch 大小
    gamma = 0.99               # 折扣因子
    epsilon = 1.0              # 初始探索率
    epsilon_min = 0.05         # 最低探索率
    epsilon_decay = 0.995      # 每回合衰减率
    target_update = 10         # 每隔 target_update 个回合更新目标网络
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # epsilon-greedy 策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # 当 replay buffer 中样本足够时，进行网络更新
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # 当前 Q 值
                q_values = policy_net(states).gather(1, actions)
                
                # 目标 Q 值：使用目标网络计算下一状态的最大 Q 值
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 衰减 epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # 每隔几个回合更新一次目标网络
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    env.close()
    # 保存训练好的模型
    torch.save(policy_net.state_dict(), "dino_dqn.pth")
    print("训练完成，模型已保存为 dino_dqn.pth")

if __name__ == "__main__":
    train()
