import gymnasium as gym
import minihack
import time

env = gym.make('MiniHack-KeyRoom-S5-v0', observation_keys=('glyphs','blstats','chars_crop'))
obs, _ = env.reset()
env.render()

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.1)
    if terminated or truncated:
        print(f"Episode ended at step {step}, reward={reward}")
        obs, _ = env.reset()
        env.render()

env.close()
