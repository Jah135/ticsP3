from collections.abc import Callable
from pathlib import Path

import ale_py
import gymnasium
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from training import train_model
from watching import watch_model, watch_model_from_file

gymnasium.register_envs(ale_py)


def prompt_list_option[T](
    prompt: str, options: list[T], fmt: Callable[[T, int], str]
) -> T:
    print(
        "\n".join(
            f"{index + 1}. {fmt(option, index)}" for index, option in enumerate(options)
        )
    )

    while True:
        try:
            index = int(input(prompt)) - 1
            return options[index]
        except IndexError:
            print("index out of bounds")
        except ValueError:
            print("invalid index number")


def create_env(is_watch: bool = False) -> VecEnv:
    return VecFrameStack(
        make_atari_env(
            "ALE/Pong-v5",
            n_envs=1,
            env_kwargs={"frameskip": 1, "render_mode": "human" if is_watch else None},
            wrapper_kwargs={"terminal_on_life_loss": False},
        ),
        n_stack=4,
    )


MODELS_DIR = (Path.cwd() / "models").resolve()


available_runs: list[Path] = []

for run_dir in MODELS_DIR.iterdir():
    if run_dir.is_dir():
        available_runs.append(run_dir)

available_runs.sort()

if input("train new model? (y/n): ").lower().strip() == "y":
    agent = train_model(create_env(), MODELS_DIR, id=f"run_{len(available_runs)}")
    watch_model(agent, create_env(True))
else:
    selected_run_dir = prompt_list_option(
        "enter run index: ", available_runs, lambda x, _: x.name
    )

    available_models: list[Path] = [
        path
        for path in selected_run_dir.iterdir()
        if path.name.endswith(".zip") and path.is_file()
    ]

    selected_model_path = prompt_list_option(
        "enter model index: ", available_models, lambda x, _: x.name
    )

    watch_model_from_file(
        selected_model_path,
        create_env(is_watch=True),
        n_steps=10000,
    )
