from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv


def watch_model(agent: DQN, env: VecEnv, n_steps: int = 2000):
    try:
        observation = env.reset()

        for _ in range(n_steps):
            action, _ = agent.predict(observation, deterministic=True)
            observation, rewards, done, infos = env.step(action)
    finally:
        env.close()


def watch_model_from_file(model_path: Path, env: VecEnv, n_steps: int = 2000):
    watch_model(DQN.load(model_path, device="cpu"), env, n_steps)
