import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import chromedriver_autoinstaller

# 自动安装 Chromedriver
chromedriver_autoinstaller.install()

# 启动 Chrome
options = webdriver.ChromeOptions()
options.add_argument("--mute-audio")  # 静音
options.add_argument("--disable-gpu")
options.add_argument("--window-size=800,600")  # 固定窗口大小
options.add_argument("--disable-blink-features=AutomationControlled")  # 防止检测自动化

driver = webdriver.Chrome(options=options)
driver.get("chrome://dino")

# 获取游戏 canvas 并点击开始游戏
canvas = driver.find_element(By.TAG_NAME, "body")
canvas.send_keys(Keys.SPACE)

# 等待游戏启动
time.sleep(1)

# 获取游戏状态的 JavaScript 代码
GET_GAME_STATE = """
var runner = Runner.instance_;
return {
    crashed: runner.crashed,
    playing: runner.playing,
    dinoY: runner.tRex.yPos,
    obstacles: runner.horizon.obstacles.map(ob => { return {x: ob.xPos, width: ob.width} })
};
"""

# AI 控制（0: 不跳，1: 跳跃）
def make_action(action):
    if action == 1:
        canvas.send_keys(Keys.SPACE)

# DQN 网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化 DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(3, 2).to(device)  # 输入：3个特征（障碍物距离、高度、恐龙 Y 位置），输出：2（跳或不跳）
# 尝试加载已训练模型
try:
    policy_net.load_state_dict(torch.load("dino_dqn.pth"))
    print("已加载训练好的模型！")
except FileNotFoundError:
    print("未找到模型文件，开始新的训练！")
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 训练参数
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
memory = []
batch_size = 64
max_memory = 10000

# 训练循环
num_episodes = 500

for episode in range(num_episodes):
    state = driver.execute_script(GET_GAME_STATE)
    if state["crashed"]:
        canvas.send_keys(Keys.SPACE)
        time.sleep(1)
    
    state = driver.execute_script(GET_GAME_STATE)
    dino_y = state["dinoY"]
    obstacles = state["obstacles"]

    if len(obstacles) > 0:
        obstacle_x = obstacles[0]["x"]
        obstacle_width = obstacles[0]["width"]
    else:
        obstacle_x, obstacle_width = 600, 0  # 默认远处无障碍物

    # 归一化状态
    obs = np.array([obstacle_x / 600, obstacle_width / 50, dino_y / 100], dtype=np.float32)

    total_reward = 0
    done = False
    while not done:
        # 选择动作（epsilon-greedy）
        if random.random() < epsilon:
            action = random.choice([0, 1])  # 随机动作
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(obs_tensor)
            action = q_values.argmax().item()

        make_action(action)
        time.sleep(0.05)

        # 获取新状态
        next_state = driver.execute_script(GET_GAME_STATE)
        next_dino_y = next_state["dinoY"]
        next_obstacles = next_state["obstacles"]

        if len(next_obstacles) > 0:
            next_obstacle_x = next_obstacles[0]["x"]
            next_obstacle_width = next_obstacles[0]["width"]
        else:
            next_obstacle_x, next_obstacle_width = 600, 0

        next_obs = np.array([next_obstacle_x / 600, next_obstacle_width / 50, next_dino_y / 100], dtype=np.float32)

        # 计算奖励
        reward = 1.0
        if next_state["crashed"]:
            reward = -100.0
            done = True

        # 经验存储
        memory.append((obs, action, reward, next_obs, done))
        if len(memory) > max_memory:
            memory.pop(0)

        obs = next_obs
        total_reward += reward

        # 训练 DQN
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = policy_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 衰减 epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    # 训练完成后保存模型
    torch.save(policy_net.state_dict(), "dino_dqn.pth")
    print("模型已保存！")


driver.quit()
