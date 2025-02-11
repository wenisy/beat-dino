from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import random
import warnings

warnings.filterwarnings('ignore', category=Warning)


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

        # 确保概率和为1
        total_priority = sum(self.priorities) + 1e-10
        probs = np.array(self.priorities) / total_priority

        # 确保batch_size不超过现有记忆数量
        batch_size = min(batch_size, len(self.memory))

        try:
            indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        except ValueError:
            # 如果出现概率问题，使用均匀采样
            indices = np.random.choice(len(self.memory), batch_size, replace=False)

        return [self.memory[i] for i in indices], indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = max(priority + 1e-6, 1e-6)

    def __len__(self):
        return len(self.memory)


class DinoAI:
    def __init__(self):
        self.memory = PrioritizedReplayBuffer(maxlen=100000)  # 增加记忆容量
        self.learning_rate = 0.001  # 提高初始学习率
        self.gamma = 0.95  # 降低折扣因子，更注重即时奖励
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 提高最小探索率
        self.epsilon_decay = 0.998  # 降低衰减速率
        self.batch_size = 64  # 减小batch size
        self.update_target_freq = 10  # 更频繁地更新目标网络
        self.scores = []
        self.avg_rewards = []
        self.losses = []
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.save_dir, 'dino_model.keras')
        self.state_path = os.path.join(self.save_dir, 'dino_ai_state.pkl')
        self.last_speed = 0
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.load_progress()

    def _build_model(self):
        model = Sequential([
            Input(shape=(9,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(2, activation='linear')
        ])
        model.compile(
            loss='mse',  # 使用MSE损失函数
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

    def load_progress(self):
        if os.path.exists(self.model_path) and os.path.exists(self.state_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
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
                print(f"成功加载已有进度: 已训练 {len(self.scores)} 回合")
            except Exception as e:
                print(f"加载进度时出错: {str(e)}，将重新开始训练")
        else:
            print("未找到进度文件，将重新开始训练")

    def preprocess_state(self, state):
        if np.all(state == 0):
            return state
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        clipped_state = np.clip(normalized_state, -3, 3)
        return clipped_state

    def get_state(self, game_state):
        try:
            if not game_state or not isinstance(game_state, dict):
                return np.zeros(9)

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

            state = np.array([
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

            state = np.clip(state, -1.0, 1.0)
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            return self.preprocess_state(state)

        except Exception as e:
            print(f"状态生成错误: {e}")
            return np.zeros(9)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([True, False])
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return bool(np.argmax(act_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0

        try:
            minibatch, indices = self.memory.sample(batch_size)
            if not minibatch:
                return 0

            states = np.array([x[0] for x in minibatch])
            next_states = np.array([x[3] for x in minibatch])

            current_q = self.model.predict(states, verbose=0)
            next_q = self.target_model.predict(next_states, verbose=0)

            X, y = [], []
            new_priorities = []

            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                target = reward
                if not done:
                    target += self.gamma * np.amax(next_q[i])
                target_f = current_q[i]
                old_val = target_f[1 if action else 0]
                target_f[1 if action else 0] = target

                td_error = abs(old_val - target)
                new_priorities.append(td_error)

                X.append(state)
                y.append(target_f)

            if X and y:
                history = self.model.fit(
                    np.array(X),
                    np.array(y),
                    epochs=1,
                    batch_size=32,
                    verbose=0
                )
                loss = history.history['loss'][0]
                self.memory.update_priorities(indices, new_priorities)
                return loss
            return 0

        except Exception as e:
            print(f"训练过程出错: {e}")
            return 0

    def update_learning_rate(self, episode):
        if episode % 100 == 0:
            new_lr = self.learning_rate * 0.95
            self.learning_rate = max(new_lr, 0.00001)
            K.set_value(self.model.optimizer.learning_rate, self.learning_rate)


def calculate_reward(state, score, last_score):
    reward = 0

    if state['crashed']:
        return -100  # 增加失败惩罚

    # 基础生存奖励
    reward += 0.5

    # 速度奖励
    speed = state.get('speed', 0)
    reward += speed * 0.1

    # 跳跃奖励
    if state['obstacles']:
        obstacle = state['obstacles'][0]
        distance = obstacle.get('x', 600)

        # 根据速度调整最佳跳跃距离
        optimal_jump_distance = 100 + speed * 3

        if state['jumping']:
            # 在最佳距离范围内跳跃给予更高奖励
            if abs(distance - optimal_jump_distance) < 30:
                reward += 10
            elif distance < 50:  # 太近跳跃
                reward -= 5
            elif distance > 200:  # 太远跳跃
                reward -= 2
        else:
            # 不跳跃时的奖励/惩罚
            if distance < 50:  # 应该跳但没跳
                reward -= 10
            elif distance > optimal_jump_distance + 50:  # 正确的不跳
                reward += 1

    # 得分奖励
    score_diff = score - last_score
    if score_diff > 0:
        reward += score_diff * 2

    return reward


def init_game():
    options = webdriver.ChromeOptions()
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=800,600")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    print("正在启动 Chrome...")
    driver.get("https://trex-runner.com/")
    time.sleep(5)

    try:
        print("正在查找游戏元素...")
        canvas = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".runner-container .runner-canvas"))
        )
        print("正在开始游戏...")
        canvas.click()
        time.sleep(2)

        ready = driver.execute_script("return typeof Runner !== 'undefined' && Runner.instance_ !== null")
        if not ready:
            print("游戏未完全加载，等待中...")
            time.sleep(3)

        actions = ActionChains(driver)
        actions.send_keys(Keys.SPACE)
        actions.perform()
        time.sleep(2)
        return driver, canvas
    except Exception as e:
        print(f"获取游戏元素失败: {e}")
        driver.quit()
        return None, None


def get_game_state(driver):
    try:
        state = driver.execute_script("""
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


def save_training_data(ai):
    scores = np.array(ai.scores)
    np.save('dino_scores.npy', scores)

    plt.figure(figsize=(15, 5))

    # Plot scores
    plt.subplot(131)
    window_size = 10
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    plt.plot(scores, alpha=0.3, label='Score')
    plt.plot(moving_avg, label=f'{window_size} Moving Avg')
    plt.title('Scores')
    plt.legend()

    # Plot average rewards
    plt.subplot(132)
    if ai.avg_rewards:
        plt.plot(ai.avg_rewards)
        plt.title('Average Rewards')

    # Plot losses
    plt.subplot(133)
    if ai.losses:
        plt.plot(ai.losses)
        plt.title('Training Loss')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


def main():
    ai = DinoAI()
    best_score = max(ai.scores) if ai.scores else 0
    episode = len(ai.scores)
    total_training_time = 0
    start_time = time.time()

    # 早期停止参数
    patience = 15
    min_delta = 5
    best_avg_score = float('-inf')
    no_improvement_counter = 0

    try:
        while True:
            episode += 1
            episode_start = time.time()
            driver, canvas = init_game()
            if not driver or not canvas:
                print("游戏初始化失败")
                return

            print(f"\n第 {episode} 回合开始...")
            actions = ActionChains(driver)
            current_score = 0
            last_score = 0
            last_state = None
            last_action = None
            episode_rewards = []
            episode_losses = []

            # 每回合的训练循环
            while True:
                try:
                    state = get_game_state(driver)
                    if not state:
                        break

                    current_state = ai.get_state(state)
                    current_score = int(''.join(map(str, state['score'])))

                    # 游戏结束处理
                    if state['crashed']:
                        if last_state is not None:
                            reward = calculate_reward(state, current_score, last_score)
                            ai.remember(last_state, last_action, reward, current_state, True)
                            episode_rewards.append(reward)

                        # 更新统计信息
                        ai.scores.append(current_score)
                        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                        avg_loss = np.mean(episode_losses) if episode_losses else 0
                        ai.avg_rewards.append(avg_reward)
                        ai.losses.append(avg_loss)

                        # 打印回合信息
                        print(f"回合 {episode} - 得分: {current_score} | 最佳: {best_score}")
                        print(f"平均奖励: {avg_reward:.2f} | 损失: {avg_loss:.4f}")

                        # 检查是否需要早期停止
                        current_avg = np.mean(ai.scores[-10:]) if len(ai.scores) >= 10 else current_score
                        if current_avg > best_avg_score + min_delta:
                            best_avg_score = current_avg
                            no_improvement_counter = 0
                        else:
                            no_improvement_counter += 1

                        if no_improvement_counter >= patience:
                            print("触发早期停止")
                            return

                        break

                    # 选择动作
                    should_jump = ai.act(current_state)
                    if should_jump:
                        actions.send_keys(Keys.SPACE).perform()

                    # 存储经验
                    if last_state is not None:
                        reward = calculate_reward(state, current_score, last_score)
                        ai.remember(last_state, last_action, reward, current_state, False)
                        episode_rewards.append(reward)

                    last_state = current_state
                    last_action = should_jump
                    last_score = current_score

                    # 训练
                    loss = ai.train(ai.batch_size)
                    if loss:
                        episode_losses.append(loss)

                    # 更新目标网络
                    if episode % ai.update_target_freq == 0:
                        ai.update_target_model()

                    time.sleep(0.01)

                except Exception as e:
                    print(f"回合执行错误: {e}")
                    break

            driver.quit()

            # 更新探索率
            ai.epsilon = max(ai.epsilon_min, ai.epsilon * ai.epsilon_decay)

            # 定期保存
            if episode % 5 == 0:
                ai.save_progress()
                save_training_data(ai)

    except KeyboardInterrupt:
        print("\n训练被中断")
        ai.save_progress()
        save_training_data(ai)


if __name__ == "__main__":
    main()
