from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Dense, Input, Dropout
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
        self.memory = deque(maxlen=20000)
        self.learning_rate = 0.0005
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9975
        self.scores = []
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
            Dropout(0.1),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2, activation='linear')
        ])
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_progress(self):
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
        try:
            # 基础检查
            if not game_state or not isinstance(game_state, dict):
                return np.zeros(9)

            # 获取障碍物信息，即使没有障碍物也返回默认值
            obstacles = game_state.get('obstacles', [])
            if obstacles:
                obstacle = obstacles[0]
                next_obstacle = obstacles[1] if len(obstacles) > 1 else {}
            else:
                # 默认障碍物设在右侧边界，宽高给个较小默认值
                obstacle = {}
                next_obstacle = {}

            try:
                obs_x = float(obstacle.get('x', 600))
            except (ValueError, TypeError):
                obs_x = 600.0

            try:
                obs_width = float(obstacle.get('width', 20))
            except (ValueError, TypeError):
                obs_width = 20.0

            try:
                obs_height = float(obstacle.get('height', 20))
            except (ValueError, TypeError):
                obs_height = 20.0

            try:
                next_x = float(next_obstacle.get('x', 600))
            except (ValueError, TypeError):
                next_x = 600.0

            try:
                speed = float(game_state.get('speed', 1))
            except (ValueError, TypeError):
                speed = 1.0

            jumping = bool(game_state.get('jumping', False))

            # 计算加速度
            try:
                acceleration = (speed - self.last_speed) / 5.0
            except (ValueError, TypeError):
                acceleration = 0.0
            self.last_speed = speed

            # 创建状态向量
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

            # 确保所有值都在合理范围内
            state = np.clip(state, -1.0, 1.0)
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

            return state

        except Exception as e:
            print(f"状态生成错误: {e}")
            return np.zeros(9)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([True, False])
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return bool(np.argmax(act_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=128):
        if len(self.memory) < batch_size:
            return

        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]

        states = np.array([x[0] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])

        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        X, y = [], []
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(next_q[i])
            target_f = current_q[i]
            target_f[1 if action else 0] = target
            X.append(state)
            y.append(target_f)

        self.model.fit(np.array(X), np.array(y), epochs=2, batch_size=32, verbose=0)


def calculate_reward(state, score, last_score):
    # 如果撞到了，给予较大的惩罚
    if state['crashed']:
        return -20

    reward = 0

    # 增加存活奖励
    reward += 0.1

    # 如果存在障碍物，根据距离和是否跳跃给予奖励或惩罚
    if state['obstacles']:
        obstacle = state['obstacles'][0]
        distance = obstacle.get('x', 600)

        if state['jumping']:
            if 80 < distance < 160:
                reward += 3
            elif 160 < distance < 250:
                reward -= 1
            elif distance < 80:
                reward -= 2
        else:
            if distance < 80:
                reward -= 2
            elif 80 < distance < 160:
                reward += 0.5
    else:
        # 如果没有障碍物，鼓励保持原位而非无谓跳跃
        if state.get('jumping', False):
            reward -= 0.5

    # 根据分数变化给予奖励
    score_diff = score - last_score
    if score_diff > 0:
        reward += score_diff * 0.5

    if state.get('speed', 0) > 10:
        reward += 0.2

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
    time.sleep(5)  # 增加等待时间
    try:
        print("正在查找游戏元素...")
        canvas = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".runner-container .runner-canvas"))
        )
        print("正在开始游戏...")
        canvas.click()
        time.sleep(2)

        # 确保游戏已经完全加载
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

        # 确保返回的状态包含所有必要的字段
        default_state = {
            'crashed': False,
            'playing': False,
            'speed': 0,
            'score': [0],
            'jumping': False,
            'obstacles': []
        }

        for key in default_state:
            if key not in state or state[key] is None:
                state[key] = default_state[key]

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
    window_size = 10
    moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(scores, alpha=0.3, label='Score')
    plt.plot(moving_avg, label=f'{window_size} Moving Avg')
    plt.title('Training Progress')
    plt.xlabel('Episodes')
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
    target_update_interval = 5  # 更频繁地更新目标网络

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
                    if not state:
                        print("无法获取游戏状态，重新开始...")
                        break

                    current_state = ai.get_state(state)
                    # 只有在游戏正在进行时才检查状态是否有效
                    if state.get('playing', False) and np.all(current_state == 0):
                        print("等待有效游戏状态...")
                        time.sleep(0.1)
                        continue
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

                        if last_state is not None:
                            reward = calculate_reward(state, current_score, last_score)
                            ai.remember(last_state, last_action, reward, current_state, True)

                        # 每回合结束后的额外训练
                        if len(ai.memory) > 1000:
                            print("执行额外训练...")
                            for _ in range(200):
                                ai.train(batch_size=128)

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
                    for _ in range(8):
                        ai.train(batch_size=128)

                    time.sleep(0.01)
                except Exception as e:
                    print(f"回合中发生错误: {e}")
                    break

            driver.quit()

            # 更新探索率，让其持续衰减到设定的最小值
            ai.epsilon = max(ai.epsilon_min, ai.epsilon * ai.epsilon_decay)

            # 更新目标网络
            if episode % target_update_interval == 0:
                ai.update_target_model()
                print("目标网络已更新")

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
