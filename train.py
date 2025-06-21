import torch
import os
import csv
import random
import datetime
import numpy as np
import math
import pandas as pd
from collections import deque

from agent import PPOAgent
from drone import Drone
from calcs import distance, clamp
import config
from fileManagement import list_existing_agents, get_agent_path, get_next_session_dir, find_latest_session_path, save_agent_config, load_agent_config


class CustomEnv:
    def __init__(self, env_width, env_height, curriculum_params, drone_size=config.DRONE_SIZE, max_episode_seconds=config.MAX_EPISODE_SECONDS):
        self.env_width = env_width
        self.env_height = env_height
        self.drone = Drone((env_width // 2, env_height // 2), drone_size, config.ENV_DIAG)
        self.update_curriculum(curriculum_params)
        self.max_episode_steps = int(max_episode_seconds * config.PHYSICS_FPS)
        self.current_episode_step = 0
        self.targets = []
        self.current_target_index = 0
        self.target_pos = None
        self.OBS_DIM = config.OBS_DIM
        self.ACT_DIM = config.ACT_DIM
        self.action_high = config.ACTION_HIGH
        self.action_low = config.ACTION_LOW
        self.TERMINATION_DELAY_STEPS_TARGET = int(config.TERMINATION_DELAY_SECONDS_TARGET * config.PHYSICS_FPS)
        self.pre_terminal_countdown = 0
        self.pre_terminal_flags = {'reached_target_pre': False}
        self.episode_rewards_components = {name: 0.0 for name in config.REWARD_COMPONENT_NAMES}
        self.observation_space = type('obs_space', (object,), {'shape': (self.OBS_DIM,)})()
        self.action_space = type('act_space', (object,), {'shape': (self.ACT_DIM,), 'high': self.action_high, 'low': self.action_low})()

        self.has_entered_landing_zone = False
        self.new_target_aiming_phase = True

    def update_curriculum(self, params):
        self.reward_weights = params
        self.min_target_x = params['min_target_spawn_x']
        self.max_target_x = params['max_target_spawn_x']
        self.min_target_y = params['min_target_spawn_y']
        self.max_target_y = params['max_target_spawn_y']
        # Get all reward weights
        for key in config.REWARD_COMPONENT_NAMES:
            if 'rings' in key or 'penalties' in key and isinstance(getattr(config, f"W_{key.upper()}_START"), list):
                self.reward_weights[key] = params.get(key, np.zeros_like(config.PROXIMITY_RADII))
            else:
                self.reward_weights[key] = params.get(key, 0.0)

    def reset(self):
        # ... (same as before) ...
        start_x = random.randint(int(self.env_width * config.MIN_SPAWN_X), int(self.env_width * config.MAX_SPAWN_X))
        start_y = random.randint(int(self.env_height * config.MIN_SPAWN_Y), int(self.env_height * config.MAX_SPAWN_Y))
        self.drone.reset(initial_pos=(start_x, start_y))
        self.targets = []
        for _ in range(config.NUM_TARGETS_PER_EPISODE):
            min_x = int(self.env_width * self.min_target_x)
            max_x = int(self.env_width * self.max_target_x)
            min_y = int(self.env_height * self.min_target_y)
            max_y = int(self.env_height * self.max_target_y)
            self.targets.append((random.randint(min_x, max_x), random.randint(min_y, max_y)))
        self.current_target_index = 0
        self.target_pos = self.targets[self.current_target_index]
        self.current_episode_step = 0
        self.pre_terminal_countdown = 0
        self.pre_terminal_flags = {'reached_target_pre': False}
        self.episode_rewards_components = {name: 0.0 for name in config.REWARD_COMPONENT_NAMES}
        self.has_entered_landing_zone = False
        self.new_target_aiming_phase = True
        return self.drone.get_observation(self.target_pos)

    def step(self, action):
        self.drone.apply_action(action, config.TIME_STEP_DELTA)
        self.drone.physics_move(config.GRAVITY, config.TIME_STEP_DELTA)
        next_observation = self.drone.get_observation(self.target_pos)
        reward, info = 0.0, {}
        current_distance = distance(self.drone.shell.center, self.target_pos)
        linear_velocity_mag = self.drone.shell.linear_velocity_magnitude
        drone_velocity_vector = self.drone.shell.vel.mean(axis=0)
        target_vector = np.array(self.target_pos) - np.array(self.drone.shell.center)
        target_distance_magnitude = np.linalg.norm(target_vector)
        target_direction_normalized = target_vector / (target_distance_magnitude + 1e-8) if target_distance_magnitude > 1e-6 else np.zeros_like(target_vector)

        reward -= config.W_TIME_PRESSURE_PENALTY
        self.episode_rewards_components['time_pressure_penalty'] -= config.W_TIME_PRESSURE_PENALTY

        landing_zone_radius = config.PROXIMITY_RADII[2]
        if not self.has_entered_landing_zone and current_distance < landing_zone_radius:
            self.has_entered_landing_zone = True

        general_av_penalty = abs(self.drone.shell.angular_velocity) * self.reward_weights['general_angular_velocity_penalty']
        reward -= general_av_penalty
        self.episode_rewards_components['general_angular_velocity_penalty'] -= general_av_penalty

        if self.has_entered_landing_zone:
            normalized_angular_vel = min(abs(self.drone.shell.angular_velocity) / (math.pi * 2), 10.0)
            braking_flip_bonus = normalized_angular_vel * self.reward_weights['braking_flip_bonus']
            reward += braking_flip_bonus
            self.episode_rewards_components['braking_flip_bonus'] += braking_flip_bonus

        if not self.has_entered_landing_zone:
            reward += np.dot(drone_velocity_vector, target_direction_normalized) * self.reward_weights['directional_speed_factor']
            self.episode_rewards_components['directional_speed_reward'] += np.dot(drone_velocity_vector, target_direction_normalized) * self.reward_weights['directional_speed_factor']

            if self.new_target_aiming_phase and linear_velocity_mag > 1.0:
                velocity_direction_normalized = drone_velocity_vector / linear_velocity_mag
                alignment_with_target = np.dot(velocity_direction_normalized, target_direction_normalized)
                if alignment_with_target > 0.95:
                    initial_aim_bonus = self.reward_weights['initial_aim_bonus']
                    reward += initial_aim_bonus
                    self.episode_rewards_components['initial_aim_bonus'] += initial_aim_bonus
                    self.new_target_aiming_phase = False

            if current_distance > 400.0 and linear_velocity_mag < 100.0:
                speed_shortfall = 100.0 - linear_velocity_mag
                penalty = (speed_shortfall / 100.0) * self.reward_weights['far_slow_speed_penalty']
                reward -= penalty
                self.episode_rewards_components['far_slow_speed_penalty'] -= penalty

            if current_distance > 250.0:
                drone_forward_x = math.cos(self.drone.shell.angle)
                drone_forward_y = math.sin(self.drone.shell.angle)
                drone_forward_vector = np.array([drone_forward_x, drone_forward_y])
                pointing_alignment = np.dot(drone_forward_vector, target_direction_normalized)
                aim_error = (1.0 - pointing_alignment)
                penalty = aim_error * self.reward_weights['aim_penalty']
                reward -= penalty
                self.episode_rewards_components['aim_penalty'] -= penalty
        else:
            min_speed = config.MANEUVER_THRESHOLDS['MIN_SPEED_FOR_MANEUVER']
            if linear_velocity_mag > min_speed:
                drone_angle = self.drone.shell.angle
                thrust_vector_x = math.cos(drone_angle - math.pi / 2)
                thrust_vector_y = math.sin(drone_angle - math.pi / 2)
                thrust_vector_normalized = np.array([thrust_vector_x, thrust_vector_y])
                alignment_factor = np.dot(target_direction_normalized, thrust_vector_normalized)
                if alignment_factor > 0:
                    speed_multiplier = 1.0 + (linear_velocity_mag / config.DRONE_MAX_SPEED_SCALAR)
                    commitment_factor = alignment_factor ** 8
                    step_point_to_brake_bonus = (commitment_factor * speed_multiplier * self.reward_weights['point_to_brake_bonus'])
                    reward += step_point_to_brake_bonus
                    self.episode_rewards_components['point_to_brake_bonus'] += step_point_to_brake_bonus

            deceleration = self.drone.shell.previous_linear_velocity_magnitude - linear_velocity_mag
            if deceleration > 5.0:
                for ringInd, radius in enumerate(config.PROXIMITY_RADII):
                    if current_distance < radius:
                        bonus_weight = self.reward_weights['braking_bonus_rings'][ringInd]
                        step_braking_bonus = deceleration * bonus_weight
                        reward += step_braking_bonus
                        self.episode_rewards_components['braking_bonus'] += step_braking_bonus
                        break

            if current_distance < 200.0 and linear_velocity_mag > 5.0:
                orbital_vector = np.array([-target_direction_normalized[1], target_direction_normalized[0]])
                orbital_speed = abs(np.dot(drone_velocity_vector, orbital_vector))
                penalty = orbital_speed * self.reward_weights['orbital_velocity_penalty']
                reward -= penalty
                self.episode_rewards_components['orbital_velocity_penalty'] -= penalty

            reward -= linear_velocity_mag * self.reward_weights['post_entry_speed_penalty']
            self.episode_rewards_components['post_entry_speed_penalty'] -= linear_velocity_mag * self.reward_weights['post_entry_speed_penalty']

        radial_velocity = np.dot(drone_velocity_vector, target_direction_normalized)
        closing_speed_bonus = -radial_velocity * self.reward_weights['final_approach_bonus']
        if closing_speed_bonus > 0:
            reward += closing_speed_bonus
            self.episode_rewards_components['final_approach_bonus'] += closing_speed_bonus

        for ringInd, radius in enumerate(config.PROXIMITY_RADII):
            if current_distance < radius:
                normalized_proximity = 1 - (current_distance / radius)
                step_proximity_bonus = normalized_proximity * config.PROXIMITY_BONUSES[ringInd]
                reward += step_proximity_bonus
                self.episode_rewards_components['proximity_bonus'] += step_proximity_bonus
                break

        upright_factor = (1 + math.cos(self.drone.shell.angle)) / 2
        speed_factor = 1.0 - min(linear_velocity_mag / config.MAX_SPEED_FOR_CONDITIONAL_UPRIGHT_BONUS, 1.0)
        dist_factor = 1.0 - min(current_distance / config.MAX_DIST_FOR_CONDITIONAL_UPRIGHT_BONUS, 1.0)
        conditional_bonus_multiplier = config.W_CONDITIONAL_UPRIGHT_BONUS_FACTOR * max(speed_factor, dist_factor)
        total_multiplier = 1.0 + conditional_bonus_multiplier
        step_upright_bonus = self.reward_weights['upright_bonus'] * upright_factor * total_multiplier
        reward += step_upright_bonus
        self.episode_rewards_components['upright_bonus'] += step_upright_bonus

        if current_distance < config.PROXIMITY_RADII[-1]:
            speed_factor_hover = 1.0 - min(linear_velocity_mag / config.PERFECT_LANDING_SPEED_THRESHOLD, 1.0)
            max_ang_vel_for_hover = math.radians(45)
            ang_vel_factor_hover = 1.0 - min(abs(self.drone.shell.angular_velocity) / max_ang_vel_for_hover, 1.0)
            step_hover_bonus = config.W_HOVER_BONUS * (speed_factor_hover * ang_vel_factor_hover) ** 2
            reward += step_hover_bonus
            self.episode_rewards_components['hover_bonus'] += step_hover_bonus

        # ... (Rest of function is identical to previous version) ...
        step_low_thrust_penalty = 0.0
        for force in self.drone.thruster_forces:
            if force < config.MIN_THRUSTER_FORCE_FOR_PENALTY:
                shortfall = config.MIN_THRUSTER_FORCE_FOR_PENALTY - force
                step_low_thrust_penalty += shortfall * config.W_LOW_THRUST_PENALTY
        reward -= step_low_thrust_penalty
        self.episode_rewards_components['low_thrust_penalty'] -= step_low_thrust_penalty
        step_stagnation_penalty = 0.0
        if current_distance > config.STAGNATION_DISTANCE_THRESHOLD:
            normalized_speed = min(linear_velocity_mag / config.STAGNATION_SPEED_THRESHOLD, 1.0)
            step_stagnation_penalty = (1.0 - normalized_speed) * self.reward_weights['stagnation_penalty']
        reward -= step_stagnation_penalty
        self.episode_rewards_components['stagnation_penalty'] -= step_stagnation_penalty
        step_boundary_penalty = 0.0
        drone_x, drone_y = self.drone.shell.center
        dist_to_left, dist_to_right = drone_x, self.env_width - drone_x
        dist_to_top, dist_to_bottom = drone_y, self.env_height - drone_y
        if dist_to_left < config.BOUNDARY_MARGIN: step_boundary_penalty += (1.0 - dist_to_left / config.BOUNDARY_MARGIN) ** 2
        if dist_to_right < config.BOUNDARY_MARGIN: step_boundary_penalty += (1.0 - dist_to_right / config.BOUNDARY_MARGIN) ** 2
        if dist_to_top < config.BOUNDARY_MARGIN: step_boundary_penalty += (1.0 - dist_to_top / config.BOUNDARY_MARGIN) ** 2
        if dist_to_bottom < config.BOUNDARY_MARGIN: step_boundary_penalty += (1.0 - dist_to_bottom / config.BOUNDARY_MARGIN) ** 2
        final_boundary_penalty = step_boundary_penalty * config.W_BOUNDARY_PENALTY
        reward -= final_boundary_penalty
        self.episode_rewards_components['boundary_penalty'] -= final_boundary_penalty
        done = False
        inner_hit_radius = min(config.PROXIMITY_RADII)
        if not (0 < drone_x < self.env_width and 0 < drone_y < self.env_height):
            done = True
            info['out_of_bounds_final'] = True
            clamped_x, clamped_y = clamp(0, self.env_width, drone_x), clamp(0, self.env_height, drone_y)
            outward_vector = np.array((drone_x, drone_y)) - np.array((clamped_x, clamped_y))
            outward_normal = outward_vector / (np.linalg.norm(outward_vector) + 1e-8)
            velocity_component_outward = np.dot(drone_velocity_vector, outward_normal)
            oob_velocity_penalty = max(0, velocity_component_outward) * self.reward_weights['oob_velocity_penalty']
            reward -= oob_velocity_penalty
            self.episode_rewards_components['oob_velocity_penalty'] -= oob_velocity_penalty
        if not done:
            if current_distance < inner_hit_radius and not self.pre_terminal_flags['reached_target_pre']:
                self.pre_terminal_flags['reached_target_pre'] = True
                self.pre_terminal_countdown = self.TERMINATION_DELAY_STEPS_TARGET
            if self.pre_terminal_countdown > 0:
                self.pre_terminal_countdown -= 1
                if self.pre_terminal_countdown <= 0:
                    if self.pre_terminal_flags['reached_target_pre']:
                        info['reached_target'] = True
                        if linear_velocity_mag < config.PERFECT_LANDING_SPEED_THRESHOLD:
                            reward += config.W_PERFECT_LANDING_BONUS
                            self.episode_rewards_components['perfect_landing_bonus'] += config.W_PERFECT_LANDING_BONUS
                        else:
                            reward -= config.W_BOUNCE_LANDING_PENALTY
                            self.episode_rewards_components['bounce_penalty'] -= config.W_BOUNCE_LANDING_PENALTY
                        self.current_target_index += 1
                        reward += config.W_TARGET_REACH_BONUS
                        self.episode_rewards_components['target_reach_bonus'] += config.W_TARGET_REACH_BONUS
                        if self.current_target_index < len(self.targets):
                            self.target_pos = self.targets[self.current_target_index]
                            self.pre_terminal_flags['reached_target_pre'] = False
                            self.pre_terminal_countdown = 0
                            self.has_entered_landing_zone = False
                            self.new_target_aiming_phase = True
                        else:
                            done = True
                            info['all_targets_reached'] = True
            self.current_episode_step += 1
            if self.current_episode_step >= self.max_episode_steps and not done:
                reward += config.W_SURVIVAL_BONUS
                self.episode_rewards_components['survival_bonus'] += config.W_SURVIVAL_BONUS
                done = True
                info['truncated'] = True
        if done:
            info['episode_rewards_components'] = self.episode_rewards_components.copy()
            info['targets_reached'] = self.current_target_index
        return next_observation, reward, done, info


if __name__ == '__main__':
    # ... (main training loop is unchanged and correct) ...
    # ... (pasting for completeness) ...
    current_obs_dim, current_act_dim, current_hidden_dims = config.OBS_DIM, config.ACT_DIM, config.hiddenDims
    selected_agent_name, agent_path, initial_epoch_offset = None, None, 0
    while True:
        mode = input("Start new agent (n) or continue training existing agent (c)? ").lower()
        if mode in ['n', 'c']:
            break
        else:
            print("Invalid input. Please enter 'n' or 'c'.")
    if mode == 'n':
        selected_agent_name = input("Enter a name for the new agent: ")
        agent_path = get_agent_path(selected_agent_name)
        if os.path.exists(agent_path) and find_latest_session_path(agent_path):
            print(f"WARNING: Agent folder '{selected_agent_name}' already exists. Starting a new session.")
        os.makedirs(agent_path, exist_ok=True)
        config_to_save = {'OBS_DIM': config.OBS_DIM, 'ACT_DIM': config.ACT_DIM, 'hiddenDims': config.hiddenDims}
        save_agent_config(agent_path, config_to_save)
        print(f"Starting new training session for agent: {selected_agent_name}")
    elif mode == 'c':
        agents = list_existing_agents()
        if not agents:
            print("No existing agents found. Please train a new agent.")
            exit()
        print("Existing agents:")
        for i, agent in enumerate(agents):
            print(f"  [{i + 1}] {agent}")
        while True:
            try:
                choice = int(input("Enter the number of the agent to continue: "))
                if 1 <= choice <= len(agents):
                    selected_agent_name = agents[choice - 1]
                    agent_path = get_agent_path(selected_agent_name)
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        try:
            loaded_config = load_agent_config(agent_path)
            current_obs_dim, current_act_dim, current_hidden_dims = loaded_config.get('OBS_DIM', config.OBS_DIM), loaded_config.get('ACT_DIM', config.ACT_DIM), tuple(loaded_config.get('hiddenDims', config.hiddenDims))
        except Exception as e:
            print(f"Error loading config for agent '{selected_agent_name}': {e}. Using default config.")
        print(f"Continuing training for agent: {selected_agent_name}")

    current_curriculum_params = {'upright_bonus': config.W_UPRIGHT_BONUS_START, 'velocity_penalty': config.W_VELOCITY_PENALTY_START, 'directional_speed_factor': config.W_DIRECTIONAL_SPEED_FACTOR_START, 'distance_penalty': config.W_DISTANCE_PENALTY_START, 'stagnation_penalty': config.W_STAGNATION_PENALTY_START, 'min_target_spawn_x': config.MIN_TARGET_SPAWN_X_START, 'max_target_spawn_x': config.MAX_TARGET_SPAWN_X_START, 'min_target_spawn_y': config.MIN_TARGET_SPAWN_Y_START,
        'max_target_spawn_y': config.MAX_TARGET_SPAWN_Y_START, 'proximity_speed_penalties': np.array(config.W_PROXIMITY_SPEED_PENALTIES_START), 'oob_velocity_penalty': config.W_OOB_VELOCITY_PENALTY_START, 'braking_bonus_rings': np.array(config.W_BRAKING_BONUS_RINGS_START), 'diff_thrust_bonus': config.W_DIFF_THRUST_BONUS_START, 'point_to_brake_bonus': config.W_POINT_TO_BRAKE_BONUS_START, 'high_speed_arrival_penalty': config.W_HIGH_SPEED_ARRIVAL_PENALTY_START,
        'lateral_velocity_penalty': config.W_LATERAL_VELOCITY_PENALTY_START, 'orbital_velocity_penalty': config.W_ORBITAL_VELOCITY_PENALTY_START, 'post_entry_speed_penalty': config.W_POST_ENTRY_SPEED_PENALTY_START, 'far_slow_speed_penalty': config.W_FAR_SLOW_SPEED_PENALTY_START, 'initial_aim_bonus': config.W_INITIAL_AIM_BONUS_START, 'braking_flip_bonus': config.W_BRAKING_FLIP_BONUS_START, 'general_angular_velocity_penalty': config.W_GENERAL_ANGULAR_VELOCITY_PENALTY_START,
        'aim_penalty': config.W_AIM_PENALTY_START, 'final_approach_bonus': config.W_FINAL_APPROACH_BONUS_START}
    recent_avg_ep_lens = deque(maxlen=30)
    max_moving_avg_len_seen = 0.0

    env = CustomEnv(config.ENV_WIDTH, config.ENV_HEIGHT, curriculum_params=current_curriculum_params, drone_size=config.DRONE_SIZE, max_episode_seconds=config.MAX_EPISODE_SECONDS)
    STEPS_PER_EPOCH = 2048
    agent_params = {'observation_space': env.observation_space, 'action_space': env.action_space, 'ac_kwargs': {'hidden_sizes': current_hidden_dims}, 'epochs': 50000, 'steps_per_epoch': STEPS_PER_EPOCH, 'max_ep_len': env.max_episode_steps, 'gamma': 0.99, 'clip_ratio': 0.2, 'pi_lr': config.PI_LR_START, 'vf_lr': config.VF_LR_START, 'train_pi_iters': 20, 'train_v_iters': 20, 'lam': 0.97, 'target_kl': 0.03, 'seed': 42}
    ppo_agent = PPOAgent(**agent_params)

    if mode == 'c':
        prev_session_path = find_latest_session_path(agent_path)
        if prev_session_path:
            latest_weights_to_load_path = os.path.join(prev_session_path, "ppo_drone_agent_weights_latest.pkl")
            if os.path.exists(latest_weights_to_load_path):
                ppo_agent.load_weights(latest_weights_to_load_path)
            else:
                print(f"WARNING: Previous weights file not found. Starting from scratch.")
        else:
            print(f"No previous sessions found. Starting from scratch.")

    current_session_to_train_path = get_next_session_dir(agent_path)
    latest_weights_file_for_saving = os.path.join(current_session_to_train_path, "ppo_drone_agent_weights_latest.pkl")
    metrics_csv_filename_for_saving = os.path.join(current_session_to_train_path, "training_metrics.csv")
    print(f"Current training run logs will be saved in: {current_session_to_train_path}")

    csv_fieldnames = ['epoch', 'steps', 'avg_episode_return', 'avg_episode_length', 'avg_targets_reached', 'interpolation_factor', 'pi_lr', 'vf_lr', 'loss_pi', 'loss_v', 'approx_kl', 'entropy', 'clipfrac'] + config.REWARD_COMPONENT_NAMES
    with open(metrics_csv_filename_for_saving, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        o, ep_ret, ep_len = env.reset(), 0, 0

        for epoch in range(initial_epoch_offset, agent_params['epochs']):
            ep_rets_in_epoch, ep_lens_in_epoch, ep_targets_in_epoch = [], [], []
            epoch_rewards_components_sums = {name: 0.0 for name in config.REWARD_COMPONENT_NAMES}

            for t in range(STEPS_PER_EPOCH):
                a, v, logp_a = ppo_agent.get_action(o)
                next_o, r, done, info = env.step(a)
                ep_ret += r
                ep_len += 1
                ppo_agent.buf.store(o, a, r, v, logp_a)
                o = next_o
                timeout = ep_len == agent_params['max_ep_len']
                terminal = done or timeout

                if terminal:
                    ep_rets_in_epoch.append(ep_ret)
                    ep_lens_in_epoch.append(ep_len)
                    ep_targets_in_epoch.append(info.get('targets_reached', 0))
                    if 'episode_rewards_components' in info:
                        for name, value in info['episode_rewards_components'].items():
                            if name in epoch_rewards_components_sums:
                                epoch_rewards_components_sums[name] += value
                    o, ep_ret, ep_len = env.reset(), 0, 0

            if not terminal:
                _, v, _ = ppo_agent.actor_critic.step(o)
                ppo_agent.buf.finish_path(v)

                if not ep_rets_in_epoch:
                    print(f"  (Epoch ended mid-episode. Logging partial episode data.)")
                    ep_rets_in_epoch.append(ep_ret)
                    ep_lens_in_epoch.append(ep_len)
                    ep_targets_in_epoch.append(env.current_target_index)
                    for name, value in env.episode_rewards_components.items():
                        if name in epoch_rewards_components_sums:
                            epoch_rewards_components_sums[name] += value
            else:
                ppo_agent.buf.finish_path(0)

            metrics = ppo_agent.update()

            avg_ep_len = np.mean(ep_lens_in_epoch)
            recent_avg_ep_lens.append(avg_ep_len)
            moving_avg_len = np.mean(recent_avg_ep_lens)
            max_moving_avg_len_seen = max(max_moving_avg_len_seen, moving_avg_len)
            progress = (max_moving_avg_len_seen - config.CURRICULUM_START_AVG_LEN) / (config.CURRICULUM_END_AVG_LEN - config.CURRICULUM_START_AVG_LEN)
            interpolation_factor = np.clip(progress, 0.0, 1.0)

            keys_to_interpolate = ['upright_bonus', 'velocity_penalty', 'directional_speed_factor', 'distance_penalty', 'stagnation_penalty', 'oob_velocity_penalty', 'diff_thrust_bonus', 'point_to_brake_bonus', 'high_speed_arrival_penalty', 'lateral_velocity_penalty', 'orbital_velocity_penalty', 'post_entry_speed_penalty', 'far_slow_speed_penalty', 'initial_aim_bonus', 'braking_flip_bonus', 'general_angular_velocity_penalty', 'aim_penalty', 'final_approach_bonus']
            for key in keys_to_interpolate:
                start_val = getattr(config, f"W_{key.upper()}_START")
                end_val = getattr(config, f"W_{key.upper()}_END")
                current_curriculum_params[key] = np.interp(interpolation_factor, [0, 1], [start_val, end_val])

            for axis in ['x', 'y']:
                for bound in ['min', 'max']:
                    key = f"{bound}_target_spawn_{axis}"
                    start_val = getattr(config, f"{key.upper()}_START")
                    end_val = getattr(config, f"{key.upper()}_END")
                    current_curriculum_params[key] = np.interp(interpolation_factor, [0, 1], [start_val, end_val])

            for key in ['proximity_speed_penalties', 'braking_bonus_rings']:
                start_val = np.array(getattr(config, f"W_{key.upper()}_START"))
                end_val = np.array(getattr(config, f"W_{key.upper()}_END"))
                current_curriculum_params[key] = start_val + (end_val - start_val) * interpolation_factor

            env.update_curriculum(current_curriculum_params)

            new_pi_lr = np.interp(interpolation_factor, [0, 1], [config.PI_LR_START, config.PI_LR_END])
            new_vf_lr = np.interp(interpolation_factor, [0, 1], [config.VF_LR_START, config.VF_LR_END])
            ppo_agent.update_learning_rates(new_pi_lr, new_vf_lr)

            avg_ep_ret = np.mean(ep_rets_in_epoch)
            avg_targets_reached = np.mean(ep_targets_in_epoch)

            num_steps_in_epoch = STEPS_PER_EPOCH

            csv_row = {'epoch': epoch, 'steps': num_steps_in_epoch, 'avg_episode_return': avg_ep_ret, 'avg_episode_length': avg_ep_len, 'avg_targets_reached': avg_targets_reached, 'interpolation_factor': interpolation_factor, 'pi_lr': new_pi_lr, 'vf_lr': new_vf_lr, 'loss_pi': metrics['loss_pi'], 'loss_v': metrics['loss_v'], 'approx_kl': metrics['approx_kl'], 'entropy': metrics['ent'], 'clipfrac': metrics['clipfrac']}
            for name in config.REWARD_COMPONENT_NAMES:
                csv_row[name] = epoch_rewards_components_sums.get(name, 0.0) / num_steps_in_epoch
            writer.writerow(csv_row)
            csvfile.flush()

            print(f"Epoch {epoch} | AvgLen={avg_ep_len:.1f} | AvgRet={avg_ep_ret:.1f} | AvgTargets={avg_targets_reached:.2f} | MaxMovingSteps={max_moving_avg_len_seen:.1f} | Curriculum={interpolation_factor:.3f}")
            if (epoch + 1) % 50 == 0:
                print(f"  LRs: pi={new_pi_lr:.2e}, vf={new_vf_lr:.2e}")
                print(f"  PointToBrake: {current_curriculum_params['point_to_brake_bonus']:.3f} | DiffThrust: {current_curriculum_params['diff_thrust_bonus']:.3f}")

            if (epoch + 1) % 5 == 0 or epoch == agent_params['epochs'] - 1:
                ppo_agent.save_weights(latest_weights_file_for_saving)
