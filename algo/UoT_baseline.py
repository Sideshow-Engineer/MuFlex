from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================
SIM_DAYS = 1
START_DATE = 201
STEP_SIZE = 300                  # 5 min = 300 s, 288 steps/day
ACTION_TYPE = "continuous"
INCLUDE_HOUR = True
REWARD_MODE = "UoT_reward"
SAVE_RESULTS = True
POWER_LIMIT = 4500.0

# Baseline controller: fixed physical action for all steps
PHYSICAL_ACTION = [25, 25, 25, 25, 25, 15]

# ============================================================
# Project path setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_wrapper import MuFlex

FMU_PATH = PROJECT_ROOT / "models" / "UoT_train" / "office_baseline.fmu"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("FMU_PATH:", FMU_PATH)

# ============================================================
# Helper functions
# ============================================================
def steps_per_day(step_size: int) -> int:
    return int(24 * 3600 // step_size)

def step_from_hour(hour_float: float, step_size: int) -> int:
    """Return 1-based step index corresponding to a clock time."""
    seconds = hour_float * 3600.0
    return int(seconds // step_size)

# ============================================================
# Step 1 - Create environment
# ============================================================
fmu_configs = [{"io_type": "OfficeS", "path": str(FMU_PATH)}]

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

obs, info = env.reset()
print("Reset done. Observation shape:", np.shape(obs))

# ============================================================
# Step 2 - Baseline controller: fixed action
# ============================================================
mins = np.asarray(env.base_mins_list[0], dtype=np.float32)
maxs = np.asarray(env.base_maxs_list[0], dtype=np.float32)
physical_action = np.asarray(PHYSICAL_ACTION, dtype=np.float32)

action = (physical_action - mins) / (maxs - mins) * 2.0 - 1.0
action = np.clip(action, -1.0, 1.0).astype(np.float32)

print("Physical action:", PHYSICAL_ACTION)
print("Normalized action ([-1, 1]):", action)

# ============================================================
# Step 3 - Interaction loop
# ============================================================
done = False
total_reward = 0.0
steps = 0

try:
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        steps += 1
finally:
    run_folder = Path(env.output_folder)
    env.close()

print(f"Episode finished. Steps={steps}, Total Reward={total_reward:.4f}")
print("Run folder:", run_folder.resolve())

# ============================================================
# Step 4 - Load saved results
# ============================================================
fmu_file = run_folder / "fmu_1_data.xlsx"
reward_file = run_folder / "rewards.xlsx"

if not fmu_file.exists():
    raise FileNotFoundError(f"FMU output file not found: {fmu_file}")
if not reward_file.exists():
    raise FileNotFoundError(f"Reward file not found: {reward_file}")

df_fmu = pd.read_excel(fmu_file)
df_reward = pd.read_excel(reward_file)

n_steps = steps_per_day(STEP_SIZE) * SIM_DAYS

df_fmu = df_fmu.iloc[:n_steps].copy()
df_reward = df_reward.iloc[:n_steps].copy()

# Use 1-based plotting index for readability
step_plus1 = df_fmu["Step"] + 1

# Office hours for shading: 8:00 to 18:00
office_start = step_from_hour(8.0, STEP_SIZE)
office_end = step_from_hour(18.0, STEP_SIZE)

# X ticks every 4 hours
x_tick_step = int((4 * 3600) // STEP_SIZE)
xticks = list(range(1, n_steps + 1, x_tick_step))
if xticks[-1] != n_steps:
    xticks.append(n_steps)

# Plot style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# ============================================================
# Plot 1 - HVAC power
# ============================================================
hvac_power = df_fmu["coilPower"] + df_fmu["fanPower"]

plt.figure(figsize=(10, 4.5), dpi=120)
plt.plot(step_plus1, hvac_power, linewidth=1.6, label="HVAC power")
plt.axhline(POWER_LIMIT, linestyle="--", linewidth=1.6, color="red",
            label=f"Power limit ({POWER_LIMIT:.0f} W)")
plt.xlabel("Step")
plt.ylabel("HVAC Power (W)")
plt.xlim(1, n_steps)
plt.xticks(xticks)
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# ============================================================
# Plot 2 - Zone temperatures
# ============================================================
plt.figure(figsize=(10, 4.5), dpi=120)
plt.axvspan(office_start, office_end, color="grey", alpha=0.35, label="Office hours (8:00-18:00)")
for col in ["z5Temp", "z1Temp", "z3Temp", "z2Temp", "z4Temp"]:
    if col in df_fmu.columns:
        plt.plot(step_plus1, df_fmu[col], linewidth=1.8, label=col)

plt.xlabel("Step")
plt.ylabel("Temperature (°C)")
plt.xlim(1, n_steps)
plt.xticks(xticks)
plt.ylim(20, 35)
plt.axhspan(23, 25, alpha=0.15, label="Comfort band (23-25°C)")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.show()

# ============================================================
# Plot 3 - Total reward
# ============================================================
plt.figure(figsize=(10, 4.5), dpi=120)
reward_step = df_reward["Step"] + 1
plt.plot(reward_step, df_reward["TotalReward"], linewidth=2.0, label="Total reward")

plt.xlabel("Step")
plt.ylabel("Reward")
plt.xlim(1, n_steps)
plt.xticks(xticks)
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()