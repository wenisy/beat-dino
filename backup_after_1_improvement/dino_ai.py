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
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
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
# Dino 游戏环境封装（利用 Selenium 复用同一浏览器）
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
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.SPACE)
            actions.perform()
            time.sleep(2)
        except Exception as e:
            print(f"初始化游戏失败: {e}")
            self.driver.quit()

    def reset(self):
        """通过 JS 重启游戏（不关闭浏览器）"""
        try:
            self.driver.execute_script("Runner.instance_.restart();")
            time.sleep(1)
        except Exception as e:
            print(f"游戏重启失败: {e}")

    def step(self, action):
        """
        模拟一步操作：
          action == True 表示跳跃（发送空格键），否则不操作
        返回：state, reward, done, info
        """
        actions = ActionChains(self.driver)
        if action:
            actions.send_keys(Keys.SPACE).perform()
        time.sleep(0.01)  # 控制步长

        state = self.get_game_state()
        done = state.get('crashed', False)
        score_str = ''.join(map(str, state.get('score', [0])))
        score = int(score_str) if score_str.isdigit() else 0

        reward = calculate_reward(state, score)
        info = {'score': score}
        return state, reward, done, info

    def get_game_state(self):
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
# 奖励函数（改进版）
# -----------------------
def calculate_reward(state, score, prev_score=0):
    reward = 0

    # 获取最近的障碍物
    obstacles = state.get('obstacles', [])
    if obstacles:
        obstacle = obstacles[0]
        distance = float(obstacle.get('x', 600))

        # 智能体正在跳跃
        if state.get('jumping', False):
            if distance < 150:  # 障碍物近且跳跃
                reward += 10  # 正确的跳跃给予高奖励
            else:  # 没有障碍物却跳跃
                reward -= 5  # 惩罚无意义跳跃
        else:  # 智能体没有跳跃
            if distance < 100:  # 障碍物很近却不跳
                reward -= 10  # 严重惩罚

    # 撞车严重惩罚
    if state.get('crashed', False):
        reward -= 100

    return reward

# -----------------------
# Dino AI 代理（使用 Double DQN、Huber Loss、梯度裁剪，同时增加 ε 重启策略）
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
        self.epsilon_decay = 0.995  # 每步衰减
        self.batch_size = 64
        self.update_target_freq = 1000  # 以步数更新目标网络

        self.memory = PrioritizedReplayBuffer(maxlen=100000)
        self.train_step = 0

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.scores = []  # 每回合得分记录
        self.avg_rewards = []  # 每回合平均奖励记录
        self.losses = []  # 每回合训练损失记录
        self.last_speed = 0

        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.save_dir, 'dino_model.keras')
        self.state_path = os.path.join(self.save_dir, 'dino_ai_state.pkl')
        self.load_progress()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        # 使用 Huber Loss 与梯度裁剪（clipnorm=1.0）
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=optimizer,
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
        if np.all(state_vector == 0):
            return state_vector
        normalized = (state_vector - np.mean(state_vector)) / (np.std(state_vector) + 1e-8)
        return np.clip(normalized, -3, 3)

    def get_state_vector(self, game_state):
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
        # 当随机数小于 ε 时，随机选择动作
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

        q_current = self.model.predict(states, verbose=0)
        q_next_online = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        X, y = [], []
        new_priorities = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
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
        if self.train_step % self.update_target_freq == 0:
            self.update_target_model()

        # 每步衰减 ε
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
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid') if len(
        scores) >= window_size else scores

    plt.subplot(131)
    plt.plot(scores, alpha=0.3, label='Score')
    plt.plot(moving_avg, label=f'{window_size} Moving Avg')
    plt.title('Scores')
    plt.legend()

    plt.subplot(132)
    if agent.avg_rewards:
        plt.plot(agent.avg_rewards)
        plt.title('Average Rewards')

    plt.subplot(133)
    if agent.losses:
        plt.plot(agent.losses)
        plt.title('Training Loss')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


# -----------------------
# 主训练流程，包含 ε 重启与动态调整学习率策略
# -----------------------
def main():
    env = DinoEnv(headless=False)
    agent = DinoAgent()
    episodes = 10000
    max_steps_per_episode = 1000

    # 用于监控最近 N 回合平均得分，用来判断是否需要重启 ε
    monitor_window = 20
    best_recent_avg = -float('inf')
    prev_loss = None  # 用于记录上一个监控窗口的平均损失

    try:
        for episode in range(len(agent.scores) + 1, episodes + 1):
            print(f"\n===== 第 {episode} 回合开始 =====")
            env.reset()
            episode_reward = 0
            episode_losses = []
            done = False
            step = 0

            game_state = env.get_game_state()
            state = agent.get_state_vector(game_state)

            while not done and step < max_steps_per_episode:
                action = agent.act(state)
                next_game_state, reward, done, info = env.step(action)
                next_state = agent.get_state_vector(next_game_state)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train()
                if loss:
                    episode_losses.append(loss)
                episode_reward += reward
                state = next_state
                step += 1
                if done:
                    break

            score = info.get('score', 0)
            agent.scores.append(score)
            avg_reward = episode_reward / step if step > 0 else 0
            agent.avg_rewards.append(avg_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            agent.losses.append(avg_loss)

            print(
                f"回合 {episode} 结束 - 得分: {score} | 平均奖励: {avg_reward:.2f} | 平均损失: {avg_loss:.4f} | 探索率: {agent.epsilon:.3f}")

            # 每 monitor_window 回合后检查最近得分的平均值和平均损失
            if episode % monitor_window == 0:
                recent_avg = np.mean(agent.scores[-monitor_window:])
                print(f"最近 {monitor_window} 回合平均得分: {recent_avg:.2f}")
                if best_recent_avg != -float('inf') and recent_avg < best_recent_avg * 0.9:
                    agent.epsilon = max(0.2, agent.epsilon * 2)
                    print("检测到性能下降，临时提升探索率!")
                else:
                    best_recent_avg = max(best_recent_avg, recent_avg)

                # 动态调整学习率
                recent_loss = np.mean(agent.losses[-monitor_window:])
                if prev_loss is not None and recent_loss > prev_loss * 1.1:
                    # 如果最近平均损失比上一个窗口高出10%，降低学习率 10%
                    new_lr = agent.learning_rate * 0.9
                    agent.learning_rate = max(new_lr, 1e-5)
                    # 直接使用 assign 方法更新优化器学习率
                    agent.model.optimizer.learning_rate.assign(agent.learning_rate)
                    print(f"降低学习率到: {agent.learning_rate:.6f}")
                prev_loss = recent_loss

            # 定期保存模型与训练数据
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
