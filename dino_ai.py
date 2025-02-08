from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle
import random


class DinoAI:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # 降低衰减速度
        self.scores = []
        self.model = self._build_model()

        # 加载之前的训练数据
        self.load_progress()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=5, activation='relu'),
            Dense(24, activation='relu'),
            Dense(2, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_progress(self):
        # 保存AI的状态
        self.model.save('dino_model.h5')
        state = {
            'memory': self.memory,
            'epsilon': self.epsilon,
            'scores': self.scores
        }
        with open('dino_ai_state.pkl', 'wb') as f:
            pickle.dump(state, f)

    def load_progress(self):
        # 加载之前的训练状态
        if os.path.exists('dino_ai_state.pkl') and os.path.exists('dino_model.h5'):
            try:
                self.model = load_model('dino_model.h5')
                with open('dino_ai_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                self.memory = state['memory']
                self.epsilon = state['epsilon']
                self.scores = state['scores']
                print(f"加载已有进度: 已训练 {len(self.scores)} 回合")
            except:
                print("无法加载之前的进度，将重新开始训练")

    def get_state(self, game_state):
        if not game_state['obstacles']:
            return np.zeros(5)

        obstacle = game_state['obstacles'][0]
        # 如果有第二个障碍物，也考虑进去
        next_obstacle = game_state['obstacles'][1] if len(game_state['obstacles']) > 1 else None

        state = np.array([
            obstacle['x'] / 600,  # 第一个障碍物距离
            obstacle['width'] / 60,  # 第一个障碍物宽度
            next_obstacle['x'] / 600 if next_obstacle else 2.0,  # 第二个障碍物距离（如果存在）
            game_state['speed'] / 13,  # 游戏速度
            float(game_state.get('jumping', False))  # 恐龙是否在跳跃
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

        # 从记忆中随机采样进行训练
        minibatch = random.sample(self.memory, 32)

        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        # 预测当前状态和下一状态的Q值
        current_q = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)

        # 准备训练数据
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

    # 添加后台运行选项
    # options.add_argument("--headless")  # 无界面模式

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

    # 计算增量奖励
    score_reward = (score - last_score) * 0.1

    # 如果有障碍物，根据跳跃决策给予奖励
    if state['obstacles']:
        obstacle = state['obstacles'][0]
        distance = obstacle['x']

        # 在合适的距离跳跃给予奖励
        if state['jumping'] and 100 < distance < 200:
            return 1 + score_reward
        # 在障碍物很远时跳跃给予惩罚
        elif state['jumping'] and distance > 300:
            return -0.5 + score_reward

    return 0.1 + score_reward  # 基础存活奖励


def save_training_data(ai):
    scores = np.array(ai.scores)
    np.save('dino_scores.npy', scores)
    plt.plot(scores)
    plt.title('训练进度')
    plt.xlabel('回合')
    plt.ylabel('得分')
    plt.savefig('training_progress.png')
    plt.close()


def main():
    ai = DinoAI()
    best_score = max(ai.scores) if ai.scores else 0
    episode = len(ai.scores)

    try:
        while True:
            episode += 1
            driver, canvas = init_game()
            if not driver or not canvas:
                print("游戏初始化失败")
                return

            print(f"第 {episode} 回合开始...")
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
                        if current_score > best_score:
                            best_score = current_score
                            print(f"新记录！得分：{current_score}")
                        else:
                            print(f"游戏结束，得分：{current_score}，最佳记录：{best_score}")

                        if last_state is not None:
                            reward = calculate_reward(state, current_score, last_score)
                            ai.remember(last_state, last_action, reward, current_state, True)

                        # 每回合都保存进度
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

                    ai.train()

                    time.sleep(0.01)

                except Exception as e:
                    print(f"发生错误: {e}")
                    break

            driver.quit()

            # 更新探索率
            ai.epsilon = max(ai.epsilon_min, ai.epsilon * ai.epsilon_decay)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
        print("保存进度...")
        ai.save_progress()
        save_training_data(ai)
        print("进度已保存")


if __name__ == "__main__":
    main()
