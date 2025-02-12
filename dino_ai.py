import os
import cv2
import time
import pickle
import random
import warnings
import numpy as np
from PIL import ImageGrab
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
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input, BatchNormalization, Dropout
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
# Dino 游戏环境封装
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

    def get_game_screenshot(self):
        """获取游戏区域的截图"""
        try:
            # 获取canvas元素的位置和大小
            canvas = self.driver.find_element(By.CSS_SELECTOR, ".runner-canvas")
            location = canvas.location
            size = canvas.size

            # 计算截图区域
            left = location['x']
            top = location['y']
            right = left + size['width']
            bottom = top + size['height']

            # 截取游戏区域
            screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
            screenshot = np.array(screenshot)
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (84, 84))
            normalized = resized / 255.0
            return normalized
        except Exception as e:
            print(f"截图失败: {e}")
            return np.zeros((84, 84))

    def step(self, action):
        """执行动作并返回结果"""
        actions = ActionChains(self.driver)
        if action:
            actions.send_keys(Keys.SPACE).perform()
        time.sleep(0.01)

        # 获取游戏状态
        screenshot = self.get_game_screenshot()
        game_state = self.get_game_state()

        # 计算奖励
        done = game_state.get('crashed', False)
        score_str = ''.join(map(str, game_state.get('score', [0])))
        score = int(score_str) if score_str.isdigit() else 0
        reward = self.calculate_reward(game_state, score)

        return screenshot, reward, done, {'score': score, 'game_state': game_state}

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
            return state if state else {
                'crashed': False,
                'playing': False,
                'speed': 0,
                'score': [0],
                'jumping': False,
                'obstacles': []
            }
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

    def calculate_reward(self, state, score):
        """计算奖励"""
        if state.get('crashed', False):
            return -100

        reward = 0
        obstacles = state.get('obstacles', [])

        if obstacles:
            obstacle = obstacles[0]
            distance = float(obstacle.get('x', 600))
            if state.get('jumping', False):
                if distance < 150:  # 障碍物近且跳跃
                    reward += 10
                else:  # 无意义跳跃
                    reward -= 5
            else:  # 没有跳跃
                if distance < 100:  # 障碍物近却不跳
                    reward -= 10

        return reward

    def close(self):
        self.driver.quit()


# -----------------------
# Dino Vision AI 代理
# -----------------------
class DinoVisionAgent:
    def __init__(self):
        self.state_size = (84, 84, 1)
        self.action_size = 2
        self.memory = PrioritizedReplayBuffer(maxlen=10000)

        # 超参数
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.train_start = 1000
        self.update_target_freq = 1000

        # 创建模型
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # 训练数据记录
        self.scores = []
        self.episodes = []
        self.average = []
        self.train_step = 0

    def _build_model(self):
        model = Sequential([
            Conv2D(32, (8, 8), strides=4, activation='relu',
                   input_shape=self.state_size),
            Conv2D(64, (4, 4), strides=2, activation='relu'),
            Conv2D(64, (3, 3), strides=1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='huber_loss',
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.choice([True, False])
        state = state.reshape((1, *self.state_size))
        act_values = self.model.predict(state, verbose=0)
        return bool(np.argmax(act_values[0]))

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.train_start:
            return 0

        minibatch, indices = self.memory.sample(self.batch_size)
        if not minibatch:
            return 0

        states = np.array([exp[0] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])

        # 预测当前状态和下一状态的Q值
        current_qs = self.model.predict(states, verbose=0)
        next_qs = self.target_model.predict(next_states, verbose=0)

        # 计算目标Q值
        targets = []
        errors = []
        for i, (state, action, reward, _, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_qs[i])

            target_f = current_qs[i]
            old_val = target_f[1 if action else 0]
            target_f[1 if action else 0] = target

            targets.append(target_f)
            errors.append(abs(old_val - target))

        # 更新优先级
        self.memory.update_priorities(indices, errors)

        # 训练网络
        history = self.model.fit(
            states,
            np.array(targets),
            batch_size=self.batch_size,
            verbose=0
        )
        loss = history.history['loss'][0]

        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def load(self, name):
        """加载模型"""
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        """保存模型"""
        self.model.save_weights(name)


# -----------------------
# 训练过程
# -----------------------
def train():
    env = DinoEnv(headless=False)  # 设置为False以查看游戏画面
    agent = DinoVisionAgent()
    episodes = 1000
    batch_size = 32

    try:
        for e in range(episodes):
            env.reset()
            state = env.get_game_screenshot()
            state = state.reshape(*agent.state_size)

            done = False
            score = 0

            while not done:
                # 选择动作
                action = agent.act(state)

                # 执行动作
                next_state, reward, done, info = env.step(action)
                next_state = next_state.reshape(*agent.state_size)

                # 存储经验
                agent.remember(state, action, reward, next_state, done)

                # 训练网络
                loss = agent.replay()

                state = next_state
                score = info['score']

                if done:
                    print(f"episode: {e}/{episodes}, score: {score}, e: {agent.epsilon:.2}")
                    agent.scores.append(score)
                    agent.episodes.append(e)
                    agent.average.append(sum(agent.scores[-10:]) / len(agent.scores[-10:]))

                    # 保存模型
                    if e % 50 == 0:
                        agent.save(f"dino_vision_{e}.weights.h5")
                    break

            # 绘制训练进度
            if e % 10 == 0:
                plt.figure(figsize=(15, 5))
                plt.subplot(131)
                plt.title('score')
                plt.plot(agent.scores)
                plt.subplot(132)
                plt.title('average')
                plt.plot(agent.average)
                plt.savefig('training_progress.png')
                plt.close()

    except KeyboardInterrupt:
        print("训练中断")
    finally:
        env.close()


if __name__ == "__main__":
    train()