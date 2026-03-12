import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed

# ============================================================
# User settings
# ============================================================
SIM_DAYS = 1
START_DATE = 201
STEP_SIZE = 300                    # 5 min = 300 s, 288 steps/day
ACTION_TYPE = "continuous"
INCLUDE_HOUR = True
REWARD_MODE = "UoT_reward"
SAVE_RESULTS = False
SEED = 42

TOTAL_TIMESTEPS = 120_000
LOG_INTERVAL = 5
TB_RUN_NAME = "SAC_UoT_office"

LEARNING_RATE = 3e-4
BUFFER_SIZE = 200_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
VERBOSE = 1

# ============================================================
# Project path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_wrapper import MuFlex

# ============================================================
# Paths
# ============================================================
FMU_PATH = PROJECT_ROOT / "models" / "UoT_train" / "office_baseline.fmu"

LOG_DIR = PROJECT_ROOT / "RL_Log"
TB_LOG_DIR = LOG_DIR / "tb"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("FMU_PATH:", FMU_PATH)
print("LOG_DIR:", LOG_DIR)
print("TB_LOG_DIR:", TB_LOG_DIR)

# TensorBoard command:
# tensorboard --logdir "<your_project_path>\\RL_Log\\tb"

# ============================================================
# Environment factory for SB3
# ============================================================
def make_env(fmu_configs, seed):
    def _init():
        env = MuFlex(
            fmu_configs=fmu_configs,
            sim_days=SIM_DAYS,
            start_date=START_DATE,
            step_size=STEP_SIZE,
            action_type=ACTION_TYPE,
            include_hour=INCLUDE_HOUR,
            reward_mode=REWARD_MODE,
            save_results=SAVE_RESULTS,
        )
        return env

    set_random_seed(seed)
    return _init

# ============================================================
# Main
# ============================================================
def main():
    # Define FMU config
    fmu_configs = [{"io_type": "OfficeS", "path": str(FMU_PATH)}]

    # Create one environment instance for checking spaces
    env = make_env(fmu_configs, seed=SEED)()

    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        verbose=VERBOSE,
        seed=SEED,
        tensorboard_log=str(TB_LOG_DIR),
    )

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=LOG_INTERVAL,
        tb_log_name=TB_RUN_NAME,
    )

    # Save trained model
    save_path = LOG_DIR / TB_RUN_NAME
    model.save(save_path)
    print(f"Model saved to: {save_path}.zip")

    # Close env
    env.close()
    print("Training finished.")

if __name__ == "__main__":
    main()