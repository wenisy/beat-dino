from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
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


class DinoAI:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.scores = []
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        # 使用 .keras 扩展名
        self.model_path = os.path.join(self.save_dir, 'dino_model.keras')
        self.state_path = os.path.join(self.save_dir, 'dino_ai_state.pkl')
        self.model = self._build_model()

        # 加载之前的训练数据
        self.load_progress()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(5,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='linear'))
        # 将 'mse' 改为 tf.keras.losses.MeanSquaredError()
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def save_progress(self):
        # 保存模型
        tf.keras.models.save_model(self.model, self.model_path)
        state = {
            'memory': self.memory,
            'epsilon': self.epsilon,
            'scores': self.scores
        }
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)

    def load_progress(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.state_path):
                try:
                    self.model = tf.keras.models.load_model(self.model_path)
                    with open(self.state_path, 'rb') as f:
                        state = pickle.load(f)
                    self.memory = state['memory']
                    self.epsilon = state['epsilon']
                    self.scores = state['scores']
                    print(f"成功加载已有进度: 已训练 {len(self.scores)} 回合")
                except Exception as e:
                    print(f"加载进度时出错: {str(e)}")
                    print("将重新开始训练")
            else:
                print(f"未找到进度文件 ({self.model_path} 或 {self.state_path})")
                print("将重新开始训练")
        except Exception as e:
            print(f"加载进度时发生异常: {str(e)}")
            print("将重新开始训练")

    def get_state(self, game_state):
        if not game_state['obstacles']:
            return np.zeros(5)

        obstacle = game_state['obstacles'][0]
        next_obstacle = game_state['obstacles'][1] if len(game_state['obstacles']) > 1 else None

        state = np.array([
            obstacle['x'] / 600,
            obstacle['width'] / 60,
            next_obstacle['x'] / 600 if next_obstacle else 2.0,
            game_state['speed'] / 13,
            float(game_state.get('jumping', False))
        ])
        return state

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([True, False])

        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return bool(np.argmax(act_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 32:
            return

        minibatch = random.sample(self.memory, 32)

        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        current_q = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)

        X = []
        y = []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q[i])

            target_f = current_q[i]
            target_f[1 if action else 0] = target

            X.append(state)
            y.append(target_f)

        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)


def init_game():
    options = webdriver.ChromeOptions()
    options.add_argument("--mute-audio")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=800,600")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    print("正在启动 Chrome...")
    driver = webdriver.Chrome(options=options)

    print("正在加载恐龙游戏...")
    driver.get("https://chromedino.com/")
    time.sleep(3)

    try:
        print("正在查找游戏元素...")
        canvas = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".runner-container .runner-canvas"))
        )

        print("正在开始游戏...")
        canvas.click()
        time.sleep(1)

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
    state = driver.execute_script("""
        return {
            crashed: Runner.instance_.crashed,
            playing: Runner.instance_.playing,
            speed: Runner.instance_.currentSpeed,
            score: Runner.instance_.distanceMeter.digits,
            jumping: Runner.instance_.tRex.jumping,
            obstacles: Runner.instance_.horizon.obstacles.map(o => ({
                x: o.xPos,
                y: o.yPos,
                width: o.width,
                height: o.height,
                type: o.typeConfig.type
            }))
        }
    """)
    return state


def calculate_reward(state, score, last_score):
    if state['crashed']:
        return -10

    reward = 0
    if state['obstacles']:
        obstacle = state['obstacles'][0]
        distance = obstacle['x']

        if state['jumping']:
            # 完美时机的跳跃
            if 100 < distance < 200:
                reward += 2
            # 过早跳跃
            elif distance > 300:
                reward -= 1
            # 过晚跳跃
            elif distance < 50:
                reward -= 2
        # 没有跳跃但应该跳跃
        elif 50 < distance < 100:
            reward -= 1

    # 存活奖励
    reward += 0.1
    # 分数奖励
    reward += (score - last_score) * 0.2

    return reward


def save_training_data(ai):
    scores = np.array(ai.scores)
    np.save('dino_scores.npy', scores)

    # 计算移动平均
    window_size = 10
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(scores, alpha=0.3, label='原始分数')
    plt.plot(moving_avg, label=f'{window_size}回合移动平均')
    plt.title('训练进度')
    plt.xlabel('回合')
    plt.ylabel('得分')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.close()


def main():
    ai = DinoAI()
    best_score = max(ai.scores) if ai.scores else 0
    episode = len(ai.scores)
    total_training_time = 0
    start_time = time.time()

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

            while True:
                try:
                    state = get_game_state(driver)
                    current_state = ai.get_state(state)
                    current_score = int(''.join(map(str, state['score'])))

                    if state['crashed']:
                        ai.scores.append(current_score)
                        episode_time = time.time() - episode_start
                        total_training_time = time.time() - start_time

                        if current_score > best_score:
                            best_score = current_score
                            print(f"新记录！得分：{current_score}")

                        print(f"回合结束 - 得分：{current_score} | 最佳：{best_score}")
                        print(f"回合用时：{episode_time:.1f}秒 | 总训练时间：{total_training_time / 3600:.1f}小时")
                        print(f"探索率(epsilon)：{ai.epsilon:.3f} | 记忆池大小：{len(ai.memory)}")

                        if last_state is not None:
                            reward = calculate_reward(state, current_score, last_score)
                            ai.remember(last_state, last_action, reward, current_state, True)

                        ai.save_progress()
                        if episode % 10 == 0:
                            save_training_data(ai)
                        break

                    should_jump = ai.act(current_state)
                    if should_jump:
                        actions.send_keys(Keys.SPACE).perform()

                    if last_state is not None:
                        reward = calculate_reward(state, current_score, last_score)
                        ai.remember(last_state, last_action, reward, current_state, False)

                    last_state = current_state
                    last_action = should_jump
                    last_score = current_score

                    # 增加训练频率
                    for _ in range(5):
                        ai.train()

                    time.sleep(0.01)

                except Exception as e:
                    print(f"发生错误: {e}")
                    break

            driver.quit()
            ai.epsilon = max(ai.epsilon_min, ai.epsilon * ai.epsilon_decay)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
        print("保存进度...")
        ai.save_progress()
        save_training_data(ai)
        print("进度已保存")

        hours = total_training_time / 3600
        print(f"\n训练总结:")
        print(f"总训练时间: {hours:.1f}小时")
        print(f"训练回合数: {episode}")
        print(f"最高得分: {best_score}")
        print(f"平均得分: {np.mean(ai.scores):.1f}")
        print(f"最终探索率: {ai.epsilon:.3f}")


if __name__ == "__main__":
    main()
