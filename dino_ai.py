from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import Sequential, load_model, clone_model
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
        # 增大经验回放池的容量
        self.memory = deque(maxlen=10000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0  # 初始探索率设为1.0，全面探索
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # 每回合后乘以0.995，使探索率更快下降
        self.scores = []
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        # 使用 .keras 扩展名保存模型
        self.model_path = os.path.join(self.save_dir, 'dino_model.keras')
        self.state_path = os.path.join(self.save_dir, 'dino_ai_state.pkl')
        self.model = self._build_model()
        # 初始化目标网络并同步权重
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        # 加载之前的训练数据（如果存在）
        self.load_progress()

    def _build_model(self):
        # 状态维度调整为6（增加了障碍物高度）
        model = Sequential()
        model.add(Input(shape=(6,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target_model(self):
        # 将目标网络权重更新为当前模型的权重
        self.target_model.set_weights(self.model.get_weights())

    def save_progress(self):
        # 保存模型和状态（记忆池、epsilon、得分历史）
        tf.keras.models.save_model(self.model, self.model_path)
        state = {
            'memory': self.memory,
            'epsilon': self.epsilon,
            'scores': self.scores
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
                self.memory = state['memory']
                self.epsilon = state['epsilon']
                self.scores = state['scores']
                print(f"成功加载已有进度: 已训练 {len(self.scores)} 回合")
            except Exception as e:
                print(f"加载进度时出错: {str(e)}，将重新开始训练")
        else:
            print("未找到进度文件，将重新开始训练")

    def get_state(self, game_state):
        # 如果没有障碍物，返回零状态（6维）
        if not game_state['obstacles']:
            return np.zeros(6)

        # 取第一个障碍物，并用字典的 get 方法设置默认值（如果属性为 None，则用 0）
        obstacle = game_state['obstacles'][0]
        next_obstacle = game_state['obstacles'][1] if len(game_state['obstacles']) > 1 else None

        # 对各个属性做安全检查：如果属性为 None，则用默认值 0（或其他合理值）
        obs_x = obstacle.get('x') if obstacle.get('x') is not None else 0
        obs_width = obstacle.get('width') if obstacle.get('width') is not None else 0
        obs_height = obstacle.get('height') if obstacle.get('height') is not None else 0

        # 对下一个障碍物，同样处理。如果没有下一个障碍物，默认设置为 2.0（后面再归一化）
        if next_obstacle is not None:
            next_x = next_obstacle.get('x') if next_obstacle.get('x') is not None else 0
        else:
            next_x = 2.0

        # 游戏速度同理
        speed = game_state.get('speed') if game_state.get('speed') is not None else 0

        jumping = game_state.get('jumping', False)

        state = np.array([
            obs_x / 600,
            obs_width / 60,
            obs_height / 50,
            next_x / 600,  # 如果没有下一个障碍物，next_x 已经设为 2.0
            speed / 13,
            float(jumping)
        ])
        return state

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([True, False])
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return bool(np.argmax(act_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        current_q = self.model.predict(states, verbose=0)
        # 用目标网络计算下一状态的 Q 值，提高训练稳定性
        next_q = self.target_model.predict(next_states, verbose=0)
        X, y = [], []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q[i])
            target_f = current_q[i]
            # action 为布尔值：True 表示跳跃，对应索引1，False 表示不跳，对应索引0
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
    # 如果需要，可启用无头模式：options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    print("正在启动 Chrome...")
    driver.get("https://trex-runner.com/")
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
    # 通过 JavaScript 获取游戏状态信息
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
    # 当游戏撞车时返回较大的负奖励
    if state['crashed']:
        return -10
    reward = 0
    if state['obstacles']:
        obstacle = state['obstacles'][0]
        distance = obstacle['x']
        if state['jumping']:
            if 100 < distance < 200:
                reward += 2
            elif distance > 300:
                reward -= 1
            elif distance < 50:
                reward -= 2
        elif 50 < distance < 100:
            reward -= 1
    # 增加生存奖励和分数提升奖励（将分数部分奖励从0.2调为0.3）
    reward += 0.1
    reward += (score - last_score) * 0.3
    return reward


def save_training_data(ai):
    scores = np.array(ai.scores)
    np.save('dino_scores.npy', scores)
    window_size = 10
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(scores, alpha=0.3, label='Original Scores')
    plt.plot(moving_avg, label=f'{window_size}Moving Avg')
    plt.title('Progress')
    plt.xlabel('Rounds')
    plt.ylabel('Scores')
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
    target_update_interval = 10  # 每10回合更新一次目标网络
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
                    # 每步多次训练以加快学习
                    for _ in range(5):
                        ai.train(batch_size=64)
                    time.sleep(0.01)
                except Exception as e:
                    print(f"回合中发生错误: {e}")
                    break
            driver.quit()
            # 每回合后更新探索率
            ai.epsilon = max(ai.epsilon_min, ai.epsilon * ai.epsilon_decay)
            # 每隔固定回合更新目标网络
            if episode % target_update_interval == 0:
                ai.update_target_model()
                print("目标网络已更新。")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        print("保存进度...")
        ai.save_progress()
        save_training_data(ai)
        total_training_time = time.time() - start_time
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
