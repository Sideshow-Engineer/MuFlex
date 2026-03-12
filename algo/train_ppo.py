"""Minimal PPO trainer for MuFlex using the main-branch environment API."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env import IO_DEFINITIONS, MuFlex as MuFlexCore
from src.reward_registry import list_available_reward_modes


class ResettableMuFlex(gym.Env):
    """Training-only wrapper that recreates the core env on every reset."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        fmu_configs,
        sim_days: int = 1,
        start_date: int = 1,
        step_size: int = 900,
        log_level: int = 7,
        action_type: str = "continuous",
        reward_mode: str = "demand_limiting_reward",
        save_results: bool = False,
        include_hour: bool = True,
        print_step_info: bool = False,
    ):
        super().__init__()
        self.fmu_configs = fmu_configs
        self.sim_days = sim_days
        self.start_date = start_date
        self.step_size = step_size
        self.log_level = log_level
        self.action_type = action_type.lower()
        self.reward_mode = reward_mode
        self.include_hour = include_hour
        self.print_step_info = bool(print_step_info)
        self._save_results = bool(save_results)
        self._env = None

        self._input_dims_list = []
        self._output_names_list = []
        for cfg in self.fmu_configs:
            io_def = IO_DEFINITIONS[cfg["io_type"]]
            self._input_dims_list.append(list(io_def["dims"]))
            self._output_names_list.append(list(io_def["OUTPUTS"]))

        self._build_action_space()
        self._build_observation_space()

    def _build_action_space(self) -> None:
        if self.action_type == "continuous":
            total_dims = sum(len(dims) for dims in self._input_dims_list)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(total_dims,), dtype=np.float32)
            return
        if self.action_type == "discrete":
            discrete_dims = []
            for dims in self._input_dims_list:
                discrete_dims.extend(dims)
            self.action_space = spaces.MultiDiscrete(discrete_dims)
            return
        raise ValueError(f"Unsupported action_type: {self.action_type}")

    def _build_observation_space(self) -> None:
        total_output_dims = sum(len(outputs) for outputs in self._output_names_list)
        time_feature_dims = 2 if self.include_hour else 0
        observation_dim = total_output_dims + time_feature_dims
        low = np.zeros(observation_dim, dtype=np.float32)
        high = np.ones(observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(observation_dim,), dtype=np.float32)

    @property
    def save_results(self) -> bool:
        if self._env is not None:
            return self._env.save_results
        return self._save_results

    @save_results.setter
    def save_results(self, value: bool) -> None:
        self._save_results = bool(value)
        if self._env is not None:
            self._env.save_results = bool(value)

    def _make_inner_env(self):
        env = MuFlexCore(
            fmu_configs=self.fmu_configs,
            sim_days=self.sim_days,
            start_date=self.start_date,
            step_size=self.step_size,
            log_level=self.log_level,
            action_type=self.action_type,
            reward_mode=self.reward_mode,
            save_results=self._save_results,
            include_hour=self.include_hour,
        )
        if not self.print_step_info:
            env._print_step_info = lambda *args, **kwargs: None
        return env

    def reset(self, seed=None, options=None):
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
        gc.collect()
        self._env = self._make_inner_env()
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        if self._env is None:
            raise RuntimeError("ResettableMuFlex.step() called before reset().")
        return self._env.step(action)

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._env is not None and hasattr(self._env, name):
            return getattr(self._env, name)
        raise AttributeError(name)


IDE_RUN_DEFAULTS = {
    "energyplus_dir": r"C:\EnergyPlusV9.2.0",
    "fmu_path": r"models\small_office\small_control_v1.fmu",
    "io_type": "OfficeS",
    "sim_days": 1,
    "start_date": 201,
    "step_size": 900,
    "action_type": "continuous",
    "include_hour": True,
    "reward_mode": "demand_limiting_reward",
    "print_step_info": False,
    "total_timesteps": 10_000,
    "learning_rate": 3e-4,
    "n_steps": 96,
    "batch_size": 32,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
    "seed": 42,
    "device": "auto",
    "eval_episodes": 1,
    "skip_eval": True,
    "save_env_results": False,
    "run_name": "ppo_small_control_v1",
    "output_dir": "runs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PPO training on one MuFlex FMU.")
    reward_modes = list_available_reward_modes() or ["demand_limiting_reward"]
    configured_reward_mode = IDE_RUN_DEFAULTS["reward_mode"]
    if configured_reward_mode not in reward_modes:
        configured_reward_mode = reward_modes[0]

    parser.add_argument("--energyplus-dir", default=IDE_RUN_DEFAULTS["energyplus_dir"])
    parser.add_argument("--fmu-path", default=IDE_RUN_DEFAULTS["fmu_path"])
    parser.add_argument("--io-type", default=IDE_RUN_DEFAULTS["io_type"])

    parser.add_argument("--sim-days", type=int, default=IDE_RUN_DEFAULTS["sim_days"])
    parser.add_argument("--start-date", type=int, default=IDE_RUN_DEFAULTS["start_date"])
    parser.add_argument("--step-size", type=int, default=IDE_RUN_DEFAULTS["step_size"])
    parser.add_argument("--action-type", choices=["continuous", "discrete"], default=IDE_RUN_DEFAULTS["action_type"])
    parser.add_argument("--include-hour", action=argparse.BooleanOptionalAction, default=IDE_RUN_DEFAULTS["include_hour"])
    parser.add_argument("--reward-mode", choices=reward_modes, default=configured_reward_mode)
    parser.add_argument("--print-step-info", action=argparse.BooleanOptionalAction, default=IDE_RUN_DEFAULTS["print_step_info"])

    parser.add_argument("--total-timesteps", type=int, default=IDE_RUN_DEFAULTS["total_timesteps"])
    parser.add_argument("--learning-rate", type=float, default=IDE_RUN_DEFAULTS["learning_rate"])
    parser.add_argument("--n-steps", type=int, default=IDE_RUN_DEFAULTS["n_steps"])
    parser.add_argument("--batch-size", type=int, default=IDE_RUN_DEFAULTS["batch_size"])
    parser.add_argument("--gamma", type=float, default=IDE_RUN_DEFAULTS["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=IDE_RUN_DEFAULTS["gae_lambda"])
    parser.add_argument("--ent-coef", type=float, default=IDE_RUN_DEFAULTS["ent_coef"])
    parser.add_argument("--seed", type=int, default=IDE_RUN_DEFAULTS["seed"])
    parser.add_argument("--device", default=IDE_RUN_DEFAULTS["device"])

    parser.add_argument("--eval-episodes", type=int, default=IDE_RUN_DEFAULTS["eval_episodes"])
    parser.add_argument("--skip-eval", action=argparse.BooleanOptionalAction, default=IDE_RUN_DEFAULTS["skip_eval"])
    parser.add_argument("--save-env-results", action=argparse.BooleanOptionalAction, default=IDE_RUN_DEFAULTS["save_env_results"])
    parser.add_argument("--run-name", default=IDE_RUN_DEFAULTS["run_name"])
    parser.add_argument("--output-dir", default=IDE_RUN_DEFAULTS["output_dir"])
    return parser.parse_args(sys.argv[1:])


def resolve_fmu_path(project_root: Path, fmu_path_str: str) -> Path:
    fmu_path = Path(fmu_path_str).expanduser()
    if not fmu_path.is_absolute():
        fmu_path = (project_root / fmu_path).resolve()
    if not fmu_path.exists():
        raise FileNotFoundError(f"FMU not found: {fmu_path}")
    return fmu_path


def configure_energyplus_runtime(energyplus_dir: str | None) -> None:
    if not energyplus_dir:
        return

    configured_path = Path(energyplus_dir).expanduser()
    if configured_path.is_file():
        energyplus_exe = configured_path
        energyplus_bin_dir = configured_path.parent
    else:
        energyplus_bin_dir = configured_path
        energyplus_exe = configured_path / "energyplus.exe"

    if not energyplus_exe.exists():
        raise FileNotFoundError(
            f"energyplus.exe not found. Set energyplus_dir to a valid folder or exe path: {energyplus_dir}"
        )

    current_path = os.environ.get("PATH", "")
    energyplus_bin_dir_str = str(energyplus_bin_dir)
    path_parts = current_path.split(os.pathsep) if current_path else []
    if energyplus_bin_dir_str not in path_parts:
        os.environ["PATH"] = (
            energyplus_bin_dir_str + os.pathsep + current_path if current_path else energyplus_bin_dir_str
        )
    os.environ["ENERGYPLUS_EXE"] = str(energyplus_exe)
    print(f"EnergyPlus configured for this run: {energyplus_exe}")


def build_run_dir(project_root: Path, output_dir: str, run_name: str | None) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = run_name or f"ppo_run_{timestamp}"
    return project_root / output_dir / final_run_name


def make_env(args: argparse.Namespace, fmu_configs: list[dict]):
    def _init():
        env = ResettableMuFlex(
            fmu_configs=fmu_configs,
            sim_days=args.sim_days,
            start_date=args.start_date,
            step_size=args.step_size,
            action_type=args.action_type,
            reward_mode=args.reward_mode,
            save_results=args.save_env_results,
            include_hour=args.include_hour,
            print_step_info=args.print_step_info,
        )
        return Monitor(env)

    return _init


def main() -> int:
    args = parse_args()
    configure_energyplus_runtime(args.energyplus_dir)

    fmu_path = resolve_fmu_path(PROJECT_ROOT, args.fmu_path)
    fmu_configs = [{"path": str(fmu_path), "io_type": args.io_type}]
    print(f"Training FMU: {fmu_path} (io_type={args.io_type})")

    run_dir = build_run_dir(PROJECT_ROOT, args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = vars(args).copy()
    config_payload["fmu_configs"] = fmu_configs
    (run_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    print(f"Run artifacts directory: {run_dir}")

    train_env = None
    eval_env = None
    try:
        try:
            train_env = DummyVecEnv([make_env(args, fmu_configs)])
        except Exception as exc:
            err = str(exc)
            if "contains no binary for this platform" in err:
                print("FMU platform mismatch: bundled FMUs only include win64 binaries.")
                return 1
            if "energyplus" in err.lower() and "not recognized" in err.lower():
                print("EnergyPlus CLI not found. Check IDE_RUN_DEFAULTS['energyplus_dir'].")
                return 1
            raise

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            verbose=1,
            device=args.device,
        )
        model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.run_name)
        model_path = run_dir / "ppo_muflex"
        model.save(str(model_path))
        print(f"Training complete. Model saved to: {model_path}.zip")

        eval_summary_path = run_dir / "eval_summary.txt"
        if args.skip_eval:
            eval_summary_path.write_text("skipped=true\n", encoding="utf-8")
            print("Evaluation skipped (--skip-eval).")
        else:
            eval_env = DummyVecEnv([make_env(args, fmu_configs)])
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
            eval_summary = (
                f"mean_reward={mean_reward:.6f}\n"
                f"std_reward={std_reward:.6f}\n"
                f"eval_episodes={args.eval_episodes}\n"
            )
            eval_summary_path.write_text(eval_summary, encoding="utf-8")
            print(f"Evaluation mean reward: {mean_reward:.6f} +/- {std_reward:.6f}")
        return 0
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
