"""Copied from ``demand_limiting_reward.py`` but reads reward hyperparameters
from the environment when the training code provides them.
"""

DEFAULT_HVAC_WEIGHT = 0.5
DEFAULT_TEMP_WEIGHT = 0.1
DEFAULT_MAX_POWER_WEIGHT = 2.0
DEFAULT_MAX_TOTAL_HVAC_POWER = 103_500.0


def green_print(*args, **kwargs):
    """Print helper that highlights debug lines in green."""
    print("\033[32m", *args, "\033[0m", **kwargs)


def _reward_params(self):
    return (
        float(getattr(self, "max_total_hvac_power", DEFAULT_MAX_TOTAL_HVAC_POWER)),
        float(getattr(self, "hvac_weight", DEFAULT_HVAC_WEIGHT)),
        float(getattr(self, "temp_weight", DEFAULT_TEMP_WEIGHT)),
        float(getattr(self, "max_power_weight", DEFAULT_MAX_POWER_WEIGHT)),
    )


def compute_reward(self, scaled_observation):
    """Demand-limiting reward with trainer-overridable hyperparameters."""
    (
        max_total_hvac_power,
        hvac_weight,
        temp_weight,
        max_power_weight,
    ) = _reward_params(self)

    hvac_penalty_value = 0.0
    temperature_penalty_value = 0.0
    max_power_penalty_value = 0.0
    raw_observation = scaled_observation * (self.obs_high - self.obs_low) + self.obs_low

    current_time_val = (
        self.time_steps[self.current_step + 1]
        if self.current_step + 1 < self.num_steps
        else self.time_steps[-1]
    )
    time_of_day = (current_time_val - self.start_time_seconds) % (24 * 3600)
    hour_now = time_of_day // 3600
    minute_now = (time_of_day % 3600) // 60

    is_work_time = (hour_now > 8 or (hour_now == 8 and minute_now >= 15)) and (
        hour_now < 18 or (hour_now == 18 and minute_now == 0)
    )
    grid_service_time = is_work_time

    index_offset = 2 if getattr(self, "include_hour", False) else 0
    total_hvac_power_actual = 0.0
    for fmu_index, cfg in enumerate(self.fmu_configs):
        outputs = self.output_names[fmu_index]
        out_len = len(outputs)
        fmu_raw_vals = raw_observation[index_offset : index_offset + out_len]
        index_offset += out_len
        io_type = cfg["io_type"]
        if io_type == "OfficeS":
            coil_idx = outputs.index("coilPower")
            fan_idx = outputs.index("fanPower")
            total_hvac_power_actual += fmu_raw_vals[coil_idx] + fmu_raw_vals[fan_idx]
        elif io_type == "OfficeM":
            coil_bot_idx = outputs.index("coilPower_bot")
            coil_mid_idx = outputs.index("coilPower_mid")
            coil_top_idx = outputs.index("coilPower_top")
            fan_bot_idx = outputs.index("fanPower_bot")
            fan_mid_idx = outputs.index("fanPower_mid")
            fan_top_idx = outputs.index("fanPower_top")
            total_hvac_power_actual += (
                fmu_raw_vals[coil_bot_idx]
                + fmu_raw_vals[coil_mid_idx]
                + fmu_raw_vals[coil_top_idx]
                + fmu_raw_vals[fan_bot_idx]
                + fmu_raw_vals[fan_mid_idx]
                + fmu_raw_vals[fan_top_idx]
            )
        else:
            continue
    hvac_penalty_value = total_hvac_power_actual / max_total_hvac_power

    if is_work_time:
        index_offset_for_temp = 2 if getattr(self, "include_hour", False) else 0
        for fmu_index, cfg in enumerate(self.fmu_configs):
            outputs = self.output_names[fmu_index]
            out_len = len(outputs)
            fmu_raw_vals = raw_observation[index_offset_for_temp : index_offset_for_temp + out_len]
            index_offset_for_temp += out_len
            temp_low, temp_high = self.temp_interval[fmu_index]
            dev_sum = 0.0
            zone_list = (
                ["z5Temp", "z1Temp", "z3Temp", "z2Temp", "z4Temp"]
                if cfg["io_type"] == "OfficeS"
                else ["BotMeanTemp", "MidMeanTemp", "TopMeanTemp"]
            )
            for z_name in zone_list:
                if z_name in outputs:
                    z_idx = outputs.index(z_name)
                    real_temp = fmu_raw_vals[z_idx]
                    dev = 0.0
                    if real_temp < temp_low:
                        dev = temp_low - real_temp
                    elif real_temp > temp_high:
                        dev = real_temp - temp_high
                    if dev > 1.0:
                        dev = dev * dev
                    dev_sum += dev
            temperature_penalty_value += (dev_sum * 100) / 10.0

    if total_hvac_power_actual > max_total_hvac_power:
        max_power_penalty_value = (2 * hvac_penalty_value) ** 2
    else:
        max_power_penalty_value = -0.5 if grid_service_time else 0.0

    self.last_hvac_penalty = hvac_penalty_value
    self.last_temp_penalty = temperature_penalty_value
    self.last_maxpower_penalty = max_power_penalty_value

    total_penalty = (
        hvac_weight * hvac_penalty_value
        + temp_weight * temperature_penalty_value
        + max_power_weight * max_power_penalty_value
    )
    final_reward = -total_penalty
    if self.current_step in [32]:
        green_print(f"[Reward] HVAC penalty: {hvac_penalty_value}")
        green_print(f"[Reward] Temperature penalty: {temperature_penalty_value}")
        green_print(f"[Reward] Max power penalty: {max_power_penalty_value}")
        green_print(f"[Reward] Total penalty: {total_penalty}")
        green_print(f"[Reward] Final Reward: {final_reward}")
    return final_reward
