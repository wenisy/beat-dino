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
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')


# -----------------------
# 改进的优先经验回放缓冲区
# -----------------------
class PrioritizedReplayBuffer:
    def __init__(self, maxlen, alpha=0.6):
        self.memory = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha

    def add(self, experience, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(experience)
        self.priorities.append(priority)

    def get_probabilities(self, beta):
        if len(self.memory) == 0:
            return [], []

        priorities = np.array(self.priorities)
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / np.sum(scaled_priorities)

        # 计算重要性采样权重
        weights = (len(self.memory) * probs) ** (-beta)
        weights = weights / np.max(weights)  # 归一化权重

        return probs, weights

    def sample(self, batch_size, beta):
        if len(self.memory) == 0:
            return [], [], []

        batch_size = min(batch_size, len(self.memory))
        probs, weights = self.get_probabilities(beta)
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        sample_weights = weights[indices]
        return samples, indices, sample_weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = max(priority + 1e-6, 1e-6)

    def __len__(self):
        return len(self.memory)


# -----------------------
# Dino游戏环境
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
        print("正在启动Chrome并加载游戏...")
        self.driver.get("https://trex-runner.com/")
        time.sleep(5)
        try:
            print("等待游戏元素加载...")
            self.canvas = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".runner-container .runner-canvas"))
            )
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
        try:
            self.driver.execute_script("Runner.instance_.restart();")
            time.sleep(1)
        except Exception as e:
            print(f"游戏重启失败: {e}")

    def step(self, action):
        actions = ActionChains(self.driver)
        if action:
            actions.send_keys(Keys.SPACE).perform()
        time.sleep(0.01)

        state = self.get_game_state()
        done = state.get('crashed', False)
        score_str = ''.join(map(str, state.get('score', [0])))
        score = int(score_str) if score_str.isdigit() else 0

        reward = self.calculate_reward(state, score)
        info = {'score': score}
        return state, reward, done, info

    def calculate_reward(self, state, score):
        """改进的奖励计算"""
        reward = 0.0
        # 基础生存奖励
        reward += 0.1
        # 基于速度的奖励
        speed = state.get('speed', 0)
        reward += speed * 0.01

        obstacles = state.get('obstacles', [])
        if obstacles:
            obstacle = obstacles[0]
            distance = float(obstacle.get('x', 600))
            if distance < 100:  # 非常近
                if state.get('jumping', False):  # 正确跳跃
                    reward += 15
                else:
                    reward -= 20
            elif distance < 200:  # 中等距离
                if state.get('jumping', False):
                    reward -= 5
                else:
                    reward += 5

        # 撞车惩罚
        if state.get('crashed', False):
            reward -= 100

        return reward

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
# Double DQN 版 Dino AI代理
# -----------------------
class DinoAgent:
    def __init__(self, state_size=10, action_size=2):
        self.state_size = state_size
        self.action_size = action_size

        # 核心超参数
        self.gamma = 0.98             # <--- 调低折扣因子
        self.learning_rate = 0.0001   # 学习率可根据需要再调小
        self.batch_size = 128         # 批大小
        self.tau = 0.01              # <--- 增大软更新系数

        # 探索相关参数
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.exploration_noise = 0.1

        # 优先经验回放参数
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.001

        # 使用单一优化器 + 梯度裁剪
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)

        # 创建模型和目标模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        # 经验回放与训练记录
        self.memory = PrioritizedReplayBuffer(maxlen=100000, alpha=self.per_alpha)
        self.train_step = 0

        # 动态奖励缩放
        self.reward_scale = 1.0
        self.reward_history = deque(maxlen=1000)

        # 性能指标记录
        self.scores = []
        self.avg_rewards = []
        self.losses = []
        self.q_values = []

        # 模型保存路径
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.save_dir, 'dino_model.keras')
        self.state_path = os.path.join(self.save_dir, 'dino_ai_state.pkl')

        self.load_progress()

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        # 第一层特征提取
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        # 残差块
        for _ in range(3):
            skip = x
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dense(128, activation='linear')(x)
            x = BatchNormalization()(x)
            x = Add()([x, skip])
            x = Activation('relu')(x)
            x = Dropout(0.1)(x)

        # 输出层
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss='huber')
        return model

    def soft_update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def update_reward_scale(self, reward):
        self.reward_history.append(abs(reward))
        if len(self.reward_history) >= 100:
            # 将动态缩放限制在 [0.2, 5.0] 之间
            scale = 1.0 / (np.mean(self.reward_history) + 1e-6)
            self.reward_scale = min(max(scale, 0.2), 5.0)

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

            # 基础特征
            obs_x = float(obstacle.get('x', 600))
            obs_width = float(obstacle.get('width', 20))
            obs_height = float(obstacle.get('height', 20))
            next_x = float(next_obstacle.get('x', 600))
            speed = float(game_state.get('speed', 1))
            jumping = 1.0 if game_state.get('jumping', False) else 0.0

            # 高级特征
            time_to_collision = obs_x / (speed + 1e-6)
            height_ratio = obs_height / 50.0
            width_ratio = obs_width / 60.0
            obstacle_density = len(obstacles) / 3.0

            state_vector = np.array([
                obs_x / 600.0,
                obs_width / 60.0,
                obs_height / 50.0,
                next_x / 600.0,
                speed / 13.0,
                jumping,
                time_to_collision / 10.0,
                height_ratio,
                width_ratio,
                obstacle_density
            ], dtype=np.float32)

            return self.preprocess_state(state_vector)
        except Exception as e:
            print(f"状态转换错误: {e}")
            return np.zeros(self.state_size)

    def act(self, state):
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return bool(np.random.choice([True, False]))

        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        self.q_values.append(np.mean(q_values))

        # 添加探索噪声
        if np.random.rand() < self.exploration_noise:
            q_values += np.random.normal(0, 0.1, size=q_values.shape)

        return bool(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
        self.update_reward_scale(reward)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        # 使用优先级采样
        minibatch, indices, weights = self.memory.sample(self.batch_size, self.per_beta)
        if not minibatch:
            return 0

        batch_size = len(minibatch)
        weights = np.array(weights, dtype=np.float32)

        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        with tf.GradientTape() as tape:
            current_q = self.model(states, training=True)
            target_q = current_q.numpy()

            # Double DQN：先用主网络选出下一步动作，再用目标网络估算Q
            # 主网络选动作
            next_actions = np.argmax(self.model(next_states, training=False), axis=1)
            # 目标网络估值
            next_q_target = self.target_model(next_states, training=False)

            # 动作掩码 (one-hot)
            action_mask = tf.one_hot(tf.cast(actions, tf.int32), self.action_size)

            for i in range(batch_size):
                if dones[i]:
                    # 回合结束
                    target_q[i][1 if actions[i] else 0] = rewards[i] * self.reward_scale
                else:
                    # Double DQN更新
                    a_next = next_actions[i]
                    q_double = next_q_target[i][a_next]
                    target_q[i][1 if actions[i] else 0] = (
                        rewards[i] * self.reward_scale + self.gamma * q_double
                    )

            losses = tf.reduce_sum(tf.square(target_q - current_q) * action_mask, axis=1)
            weights = tf.reshape(weights, [-1])
            losses = tf.reshape(losses, [-1])
            weighted_losses = tf.reduce_mean(weights * losses)

        # 计算梯度并更新
        gradients = tape.gradient(weighted_losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # 更新优先级
        td_errors = np.abs(target_q - current_q.numpy())
        priorities = np.mean(td_errors, axis=1)
        self.memory.update_priorities(indices, priorities)

        # 软更新目标网络
        self.soft_update_target_model()

        # 更新beta值
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        return float(weighted_losses.numpy())

    def save_progress(self):
        tf.keras.models.save_model(self.model, self.model_path)
        state = {
            'memory': self.memory.memory,
            'priorities': self.memory.priorities,
            'epsilon': self.epsilon,
            'scores': self.scores,
            'avg_rewards': self.avg_rewards,
            'losses': self.losses,
            'q_values': self.q_values,
            'reward_scale': self.reward_scale,
            'reward_history': list(self.reward_history)
        }
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"模型及训练状态已保存。当前ε={self.epsilon:.3f}")

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
                self.q_values = state.get('q_values', [])
                self.reward_scale = state.get('reward_scale', 1.0)
                self.reward_history = deque(state.get('reward_history', []), maxlen=1000)

                print(f"加载进度成功: 已训练{len(self.scores)}回合，当前ε={self.epsilon:.3f}")
            except Exception as e:
                print(f"加载进度出错: {e}，将重新开始训练")
        else:
            print("未找到保存的进度，将从头开始训练。")


# -----------------------
# 训练数据可视化
# -----------------------
def save_training_data(agent):
    scores = np.array(agent.scores)
    np.save('dino_scores.npy', scores)

    plt.figure(figsize=(20, 5))

    # 得分曲线
    plt.subplot(141)
    window_size = 10
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
        plt.plot(scores, alpha=0.3, label='Score')
        plt.plot(range(window_size - 1, len(scores)), moving_avg, label=f'{window_size}-MA')
    else:
        plt.plot(scores, label='Score')
    plt.title('Scores')
    plt.legend()

    # 平均奖励
    plt.subplot(142)
    if agent.avg_rewards:
        window_size = 10
        avg_r = np.array(agent.avg_rewards)
        if len(avg_r) >= window_size:
            moving_avg_r = np.convolve(avg_r, np.ones(window_size) / window_size, mode='valid')
            plt.plot(avg_r, alpha=0.3, label='Reward')
            plt.plot(range(window_size - 1, len(avg_r)), moving_avg_r, label=f'{window_size}-MA')
        else:
            plt.plot(avg_r, label='Reward')
        plt.title('Average Rewards')
        plt.legend()

    # 训练损失
    plt.subplot(143)
    if agent.losses:
        window_size = 10
        losses = np.array(agent.losses)
        if len(losses) >= window_size:
            moving_avg_l = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
            plt.plot(losses, alpha=0.3, label='Loss')
            plt.plot(range(window_size - 1, len(losses)), moving_avg_l, label=f'{window_size}-MA')
        else:
            plt.plot(losses, label='Loss')
        plt.title('Training Loss')
        plt.legend()

    # 平均Q值
    plt.subplot(144)
    if agent.q_values:
        window_size = 10
        q_vals = np.array(agent.q_values)
        if len(q_vals) >= window_size:
            moving_avg_q = np.convolve(q_vals, np.ones(window_size) / window_size, mode='valid')
            plt.plot(q_vals, alpha=0.3, label='Q-Value')
            plt.plot(range(window_size - 1, len(q_vals)), moving_avg_q, label=f'{window_size}-MA')
        else:
            plt.plot(q_vals, label='Q-Value')
        plt.title('Average Q-Values')
        plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


# -----------------------
# 主训练循环
# -----------------------
def main():
    env = DinoEnv(headless=True)
    agent = DinoAgent(state_size=10, action_size=2)
    episodes = 10000
    max_steps_per_episode = 2000

    # 训练监控参数
    monitor_window = 20
    best_reward = -float('inf')
    no_improvement_count = 0

    try:
        for episode in range(len(agent.scores) + 1, episodes + 1):
            print(f"\n===== 第 {episode} 回合开始 =====")
            env.reset()
            episode_reward = 0
            episode_losses = []
            done = False
            step = 0

            state = agent.get_state_vector(env.get_game_state())

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

            score = info.get('score', 0)
            agent.scores.append(score)
            avg_reward = episode_reward / step if step > 0 else 0
            agent.avg_rewards.append(avg_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            agent.losses.append(avg_loss)

            print(f"回合 {episode} - 得分: {score} | 步数: {step} | 平均奖励: {avg_reward:.2f} | ε: {agent.epsilon:.3f} | 奖励比例: {agent.reward_scale:.2f}")

            # 监控窗口
            if episode % monitor_window == 0:
                recent_rewards = agent.avg_rewards[-monitor_window:]
                avg_recent_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0

                if avg_recent_reward > best_reward:
                    best_reward = avg_recent_reward
                    no_improvement_count = 0
                    agent.save_progress()
                else:
                    no_improvement_count += 1

                # 如果长期没有改善，适度增加探索
                if no_improvement_count >= 5:
                    agent.epsilon = min(0.6, agent.epsilon * 1.1)
                    agent.exploration_noise *= 1.1
                    no_improvement_count = 0
                    print("检测到性能停滞，适度增加探索!")

                print(f"最近{monitor_window}回合平均奖励: {avg_recent_reward:.2f}")
                save_training_data(agent)

            # 定期保存
            if episode % 50 == 0:
                agent.save_progress()

    except KeyboardInterrupt:
        print("\n训练中断，正在保存...")
        agent.save_progress()
        save_training_data(agent)
    finally:
        env.close()


if __name__ == "__main__":
    main()
