from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle


class DinoAI:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.scores = []

        # 加载之前的训练数据
        self.load_progress()

    def save_progress(self):
        # 保存AI的状态
        state = {
            'memory': self.memory,
            'epsilon': self.epsilon,
            'scores': self.scores
        }
        with open('dino_ai_state.pkl', 'wb') as f:
            pickle.dump(state, f)

    def load_progress(self):
        # 加载之前的训练状态
        if os.path.exists('dino_ai_state.pkl'):
            try:
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
            return np.zeros(3)
        obstacle = game_state['obstacles'][0]
        return np.array([
            obstacle['x'] / 600,
            obstacle['width'] / 60,
            game_state['speed'] / 13
        ])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([True, False])
        distance = state[0] * 600
        return 100 < distance < 200

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 32:
            return


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
            last_state = None
            last_action = None

            while True:
                try:
                    state = get_game_state(driver)
                    current_state = ai.get_state(state)

                    if state['crashed']:
                        ai.scores.append(current_score)
                        if current_score > best_score:
                            best_score = current_score
                            print(f"新记录！得分：{current_score}")
                        else:
                            print(f"游戏结束，得分：{current_score}，最佳记录：{best_score}")

                        if last_state is not None:
                            ai.remember(last_state, last_action, -1, current_state, True)

                        # 每回合都保存进度
                        ai.save_progress()
                        if episode % 10 == 0:
                            save_training_data(ai)
                        break

                    should_jump = ai.act(current_state)
                    if should_jump:
                        actions.send_keys(Keys.SPACE).perform()

                    if last_state is not None:
                        reward = 1
                        ai.remember(last_state, last_action, reward, current_state, False)

                    last_state = current_state
                    last_action = should_jump

                    current_score = int(''.join(map(str, state['score'])))

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


if __name__ == "__main__":
    main()
