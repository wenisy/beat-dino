import os
import time
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore', category=Warning)


# -----------------------
# 优先经验回放缓冲区
# -----------------------
class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.memory) == 0:
            return [], []
        total_priority = sum(self.priorities) + 1e-10
        probs = np.array(self.priorities) / total_priority
        batch_size = min(batch_size, len(self.memory))
        try:
            indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        except ValueError:
            indices = np.random.choice(len(self.memory), batch_size, replace=False)
        samples = [self.memory[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = max(priority + 1e-6, 1e-6)

    def __len__(self):
        return len(self.memory)


# -----------------------
# Dino游戏环境封装（利用Selenium复用同一浏览器）
# -----------------------
class DinoEnv:
    def __init__(self, headless=True):
        options = webdriver.ChromeOptions()
        options.add_argument("--mute-audio")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=800,600")
        options.add_argument("--disable-blink-features=AutomationControlled")
        if headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        self._init_game()

    def _init_game(self):
        print("正在启动 Chrome 并加载游戏...")
        self.driver.get("https://trex-runner.com/")
        time.sleep(5)
        try:
            print("等待游戏元素加载...")
            self.canvas = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".runner-container .runner-canvas"))
            )
            # 点击启动游戏
            self.canvas.click()
            time.sleep(2)
            # 触发一次空格键启动游戏
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.SPACE)
            actions.perform()
            time.sleep(2)
        except Exception as e:
            print(f"初始化游戏失败: {e}")
            self.driver.quit()

    def reset(self):
        """使用 JS 调用 Runner.instance_.restart() 复位游戏"""
        try:
            self.driver.execute_script("Runner.instance_.restart();")
            time.sleep(1)
        except Exception as e:
            print(f"游戏重启失败: {e}")

    def step(self, action):
        """
        模拟一步操作：
          action == True：跳跃（发送空格键）
          action == False：不操作
        返回：state, reward, done, info
        """
        actions = ActionChains(self.driver)
        if action:
            actions.send_keys(Keys.SPACE).perform()
        time.sleep(0.01)  # 控制步长

        state = self.get_game_state()
        done = state.get('crashed', False)
        # 得分处理（score 为数字列表，转换为整数）
        score_str = ''.join(map(str, state.get('score', [0])))
        score = int(score_str) if score_str.isdigit() else 0

        reward = calculate_reward(state, score)
        info = {'score': score}
        return state, reward, done, info

    def get_game_state(self):
        """从浏览器中提取游戏状态"""
        try:
            state = self.driver.execute_script("""
                if (typeof Runner === 'undefined' || !Runner.instance_) {
                    return null;
                }
                var runner = Runner.instance_;
                var obstacles = [];
                try {
                    obstacles = runner.horizon.obstacles.map(function(o) {
                        return {
                            x: o.xPos || 0,
                            y: o.yPos || 0,
                            width: o.width || 0,
                            height: o.height || 0,
                            type: o.typeConfig ? o.typeConfig.type : 'unknown'
                        };
                    });
                } catch (e) {
                    obstacles = [];
                }
                return {
                    crashed: Boolean(runner.crashed),
                    playing: Boolean(runner.playing),
                    speed: Number(runner.currentSpeed) || 0,
                    score: Array.isArray(runner.distanceMeter.digits) ? runner.distanceMeter.digits : [0],
                    jumping: Boolean(runner.tRex.jumping),
                    obstacles: obstacles
                };
            """)
            if not state:
                return {
                    'crashed': False,
                    'playing': False,
                    'speed': 0,
                    'score': [0],
                    'jumping': False,
                    'obstacles': []
                }
            return state
        except Exception as e:
            print(f"获取游戏状态错误: {e}")
            return {
                'crashed': False,
                'playing': False,
                'speed': 0,
                'score': [0],
                'jumping': False,
                'obstacles': []
            }

    def close(self):
        self.driver.quit()


# -----------------------
# 奖励函数
# -----------------------
def calculate_reward(state, score, optimal_score_diff=1):
    """
    设计思路：
      - 如果游戏结束，则给出较大惩罚
      - 生存奖励 + 速度奖励
      - 如果有障碍物，则根据距离和跳跃情况给出奖励或惩罚
      - 得分差分奖励
    """
    reward = 0
    if state.get('crashed', False):
        return -100

    reward += 0.5  # 生存奖励
    speed = state.get('speed', 0)
    reward += speed * 0.1

    if state.get('obstacles'):
        obstacle = state['obstacles'][0]
        distance = float(obstacle.get('x', 600))
        optimal_jump_distance = 100 + speed * 3  # 随速度调整
        if state.get('jumping', False):
            if abs(distance - optimal_jump_distance) < 30:
                reward += 10  # 跳跃时机正确
            elif distance < 50:
                reward -= 5  # 太近跳跃惩罚
            elif distance > 200:
                reward -= 2  # 太远跳跃惩罚
        else:
            if distance < 50:
                reward -= 10  # 应该跳但没跳惩罚
            elif distance > optimal_jump_distance + 50:
                reward += 1  # 正常不跳奖励

    # 得分差分奖励（假设每步得分增加较小）
    reward += optimal_score_diff * 2
    return reward


# -----------------------
# Dino AI 代理（使用Double DQN改进）
# -----------------------
class DinoAgent:
    def __init__(self, state_size=9, action_size=2):
        self.state_size = state_size
        self.action_size = action_size

        # 超参数
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子（可以适当调慢）
        self.batch_size = 64
        self.update_target_freq = 1000  # 用步数而非回合更新目标网络

        self.memory = PrioritizedReplayBuffer(maxlen=100000)
        self.train_step = 0

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.scores = []
        self.avg_rewards = []
        self.losses = []
        self.last_speed = 0

        # 保存路径
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.save_dir, 'dino_model.keras')
        self.state_path = os.path.join(self.save_dir, 'dino_ai_state.pkl')
        self.load_progress()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_progress(self):
        tf.keras.models.save_model(self.model, self.model_path)
        state = {
            'memory': self.memory.memory,
            'priorities': self.memory.priorities,
            'epsilon': self.epsilon,
            'scores': self.scores,
            'avg_rewards': self.avg_rewards,
            'losses': self.losses
        }
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)
        print("模型及训练状态已保存。")

    def load_progress(self):
        if os.path.exists(self.model_path) and os.path.exists(self.state_path):
            try:
                self.model = load_model(self.model_path)
                self.target_model = clone_model(self.model)
                self.target_model.set_weights(self.model.get_weights())
                with open(self.state_path, 'rb') as f:
                    state = pickle.load(f)
                self.memory.memory = state['memory']
                self.memory.priorities = state['priorities']
                self.epsilon = state['epsilon']
                self.scores = state['scores']
                self.avg_rewards = state.get('avg_rewards', [])
                self.losses = state.get('losses', [])
                print(f"加载进度成功: 已训练 {len(self.scores)} 回合")
            except Exception as e:
                print(f"加载进度出错: {e}，将重新开始训练")
        else:
            print("未找到保存的进度，将从头开始训练。")

    def preprocess_state(self, state_vector):
        # 如果状态全为 0，则直接返回
        if np.all(state_vector == 0):
            return state_vector
        normalized = (state_vector - np.mean(state_vector)) / (np.std(state_vector) + 1e-8)
        return np.clip(normalized, -3, 3)

    def get_state_vector(self, game_state):
        """
        将游戏状态字典转换为长度为9的状态向量
        """
        try:
            obstacles = game_state.get('obstacles', [])
            if obstacles:
                obstacle = obstacles[0]
                next_obstacle = obstacles[1] if len(obstacles) > 1 else {}
            else:
                obstacle = {}
                next_obstacle = {}

            obs_x = float(obstacle.get('x', 600))
            obs_width = float(obstacle.get('width', 20))
            obs_height = float(obstacle.get('height', 20))
            next_x = float(next_obstacle.get('x', 600))
            speed = float(game_state.get('speed', 1))
            jumping = bool(game_state.get('jumping', False))
            acceleration = (speed - self.last_speed) / 5.0
            self.last_speed = speed

            state_vector = np.array([
                obs_x / 600.0,
                obs_width / 60.0,
                obs_height / 50.0,
                next_x / 600.0,
                speed / 13.0,
                float(jumping),
                acceleration,
                (obs_x / speed) if speed > 0 else 0.0,
                1.0 if obs_height > 40 else 0.0
            ], dtype=np.float32)

            state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            state_vector = np.clip(state_vector, -1.0, 1.0)
            return self.preprocess_state(state_vector)
        except Exception as e:
            print(f"状态转换错误: {e}")
            return np.zeros(self.state_size)

    def act(self, state_vector):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([True, False])
        q_values = self.model.predict(state_vector.reshape(1, -1), verbose=0)[0]
        return bool(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        minibatch, indices = self.memory.sample(self.batch_size)
        if not minibatch:
            return 0

        states = np.array([exp[0] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])

        # 采用 Double DQN 计算目标值
        q_current = self.model.predict(states, verbose=0)
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        X, y = [], []
        new_priorities = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # 先用在线网络选取动作，再从目标网络取值
                next_action = np.argmax(q_next_online[i])
                target += self.gamma * q_next_target[i][next_action]
            target_f = q_current[i]
            action_idx = 1 if action else 0
            old_val = target_f[action_idx]
            target_f[action_idx] = target

            td_error = abs(old_val - target)
            new_priorities.append(td_error)

            X.append(state)
            y.append(target_f)

        history = self.model.fit(
            np.array(X),
            np.array(y),
            epochs=1,
            batch_size=32,
            verbose=0
        )
        loss = history.history['loss'][0]
        self.memory.update_priorities(indices, new_priorities)

        self.train_step += 1
        # 每固定步数更新一次目标网络
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()

        # 每步衰减 epsilon（或者也可以按回合衰减）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


# -----------------------
# 绘制训练数据
# -----------------------
def save_training_data(agent):
    scores = np.array(agent.scores)
    np.save('dino_scores.npy', scores)

    plt.figure(figsize=(15, 5))
    window_size = 10
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    else:
        moving_avg = scores

    # 绘制得分曲线
    plt.subplot(131)
    plt.plot(scores, alpha=0.3, label='Score')
    plt.plot(moving_avg, label=f'{window_size} Moving Avg')
    plt.title('Scores')
    plt.legend()

    # 绘制平均奖励
    plt.subplot(132)
    if agent.avg_rewards:
        plt.plot(agent.avg_rewards)
        plt.title('Average Rewards')

    # 绘制训练损失
    plt.subplot(133)
    if agent.losses:
        plt.plot(agent.losses)
        plt.title('Training Loss')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


# -----------------------
# 主训练流程
# -----------------------
def main():
    env = DinoEnv(headless=True)
    agent = DinoAgent()
    episodes = 10000  # 设定最大回合数
    max_steps_per_episode = 1000  # 每回合最大步数（避免无限循环）

    try:
        for episode in range(len(agent.scores) + 1, episodes + 1):
            print(f"\n===== 第 {episode} 回合开始 =====")
            env.reset()  # 复位游戏，而不是重启浏览器
            episode_reward = 0
            episode_losses = []
            done = False
            step = 0

            # 获取初始状态向量
            game_state = env.get_game_state()
            state = agent.get_state_vector(game_state)

            while not done and step < max_steps_per_episode:
                action = agent.act(state)
                # 执行动作，获取下一状态及奖励
                next_game_state, reward, done, info = env.step(action)
                next_state = agent.get_state_vector(next_game_state)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train()
                if loss:
                    episode_losses.append(loss)
                episode_reward += reward
                state = next_state
                step += 1

                # 若游戏崩溃，则跳出循环
                if done:
                    break

            # 记录本回合得分（此处使用 info 中的 score）
            score = info.get('score', 0)
            agent.scores.append(score)
            avg_reward = episode_reward / step if step > 0 else 0
            agent.avg_rewards.append(avg_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            agent.losses.append(avg_loss)

            print(
                f"回合 {episode} 结束 - 得分: {score} | 平均奖励: {avg_reward:.2f} | 平均损失: {avg_loss:.4f} | 探索率: {agent.epsilon:.3f}")

            # 定期保存模型与数据
            if episode % 10 == 0:
                agent.save_progress()
                save_training_data(agent)
    except KeyboardInterrupt:
        print("\n训练中断，正在保存进度...")
        agent.save_progress()
        save_training_data(agent)
    finally:
        env.close()


if __name__ == "__main__":
    main()
