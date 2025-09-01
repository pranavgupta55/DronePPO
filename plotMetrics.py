import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import datetime
import numpy as np
import random

from fileManagement import list_existing_agents, find_sessions_for_agent
from config import REWARD_COMPONENT_NAMES

# --- UPDATED REWARD KEYS ---
POSITIVE_REWARD_KEYS = [
    'survival_bonus', 'proximity_bonus', 'directional_speed_reward', 'upright_bonus',
    'hover_bonus', 'perfect_landing_bonus', 'target_reach_bonus',
    'braking_bonus', 'diff_thrust_bonus', 'point_to_brake_bonus',
    'initial_aim_bonus', 'braking_flip_bonus', 'final_approach_bonus', 'is_alive_bonus'
]
NEGATIVE_REWARD_KEYS = [
    'boundary_penalty', 'distance_penalty', 'power_penalty',
    'velocity_penalty', 'bounce_penalty', 'vertical_velocity_penalty',
    'proximity_speed_penalty', 'low_thrust_penalty', 'oob_velocity_penalty',
    'stagnation_penalty', 'high_speed_arrival_penalty', 'lateral_velocity_penalty',
    'orbital_velocity_penalty', 'post_entry_speed_penalty',
    'far_slow_speed_penalty', 'general_angular_velocity_penalty', 'aim_penalty'
]


def plot_training_metrics(agent_name=None, base_dir="runs"):
    if agent_name is None:
        agents = list_existing_agents(base_dir)
        if not agents:
            print("No agents found to plot.")
            return
        print("Available agents:")
        for i, agent in enumerate(agents):
            print(f"  [{i + 1}] {agent}")
        while True:
            try:
                choice = int(input("Enter the number of the agent to plot: "))
                if 1 <= choice <= len(agents):
                    agent_name = agents[choice - 1]
                    break
                else:
                    print("Invalid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    agent_path = os.path.join(base_dir, agent_name)
    session_paths = find_sessions_for_agent(agent_path)

    if not session_paths:
        print(f"No training sessions found for agent '{agent_name}'.")
        return

    all_data = []
    session_epoch_ranges = []
    current_global_epoch_offset = 0

    print(f"Collecting data for agent: {agent_name}")
    for i, session_path in enumerate(session_paths):
        metrics_file = os.path.join(session_path, "training_metrics.csv")
        session_name = os.path.basename(session_path)
        print(f"  Loading session: {session_name}")
        if not os.path.exists(metrics_file):
            print(f"    Metrics file not found for session '{session_name}'.")
            continue
        try:
            df_session = pd.read_csv(metrics_file)
            if df_session.empty or 'epoch' not in df_session.columns:
                print(f"    Metrics file for session '{session_name}' is empty or missing 'epoch' column.")
                continue
            if 'steps' not in df_session.columns: df_session['steps'] = 0
            if 'episodes' not in df_session.columns: df_session['episodes'] = 0
            if 'interpolation_factor' not in df_session.columns: df_session['interpolation_factor'] = -1
            if 'avg_targets_reached' not in df_session.columns: df_session['avg_targets_reached'] = 0

            df_session['global_epoch'] = df_session['epoch'] + current_global_epoch_offset
            all_data.append(df_session)
            session_epoch_ranges.append((df_session['global_epoch'].min(), df_session['global_epoch'].max(), session_name))
            current_global_epoch_offset = df_session['global_epoch'].max() + 1
        except Exception as e:
            print(f"Error reading CSV file {metrics_file}: {e}")
            continue
    if not all_data:
        print(f"No valid data found across any sessions for agent '{agent_name}'.")
        return

    df = pd.concat(all_data, ignore_index=True)
    DOWNSAMPLE_THRESHOLD = 1000
    total_epochs = len(df)
    if total_epochs > DOWNSAMPLE_THRESHOLD:
        step = total_epochs // DOWNSAMPLE_THRESHOLD
        print(f"Data has {total_epochs} epochs. Downsampling by plotting every {step}-th epoch.")
        df_plot = df.iloc[::step].copy()
    else:
        df_plot = df

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(17, 9.5))
    gs = matplotlib.gridspec.GridSpec(4, 2, height_ratios=[1, 1, 0.8, 2.2], hspace=0.7, wspace=0.6)

    axs = {'return': fig.add_subplot(gs[0, 0]), 'loss': fig.add_subplot(gs[0, 1]), 'kl_clip': fig.add_subplot(gs[1, 0]), 'length_steps': fig.add_subplot(gs[1, 1]), 'curriculum': fig.add_subplot(gs[2, 0]), 'targets': fig.add_subplot(gs[2, 1]), 'rewards': fig.add_subplot(gs[3, :])}
    fig.suptitle(f"Training Metrics for Agent: {agent_name}", fontsize=18, fontweight='bold')

    axs['return'].plot(df_plot['global_epoch'], df_plot['avg_episode_return'], color='deepskyblue', label='Avg Return', marker='.', linestyle='-')
    axs['loss'].plot(df_plot['global_epoch'], df_plot['loss_pi'], color='crimson', label='Policy Loss')
    axs['loss'].plot(df_plot['global_epoch'], df_plot['loss_v'], color='orange', label='Value Loss')
    axs['kl_clip'].plot(df_plot['global_epoch'], df_plot['approx_kl'], color='darkviolet', label='Approx KL')
    axs['kl_clip'].plot(df_plot['global_epoch'], df_plot['clipfrac'], color='forestgreen', label='Clip Fraction')
    axs['kl_clip'].axhline(y=0.01, color='gray', linestyle='--', linewidth=1, label='Target KL')

    ax_len = axs['length_steps']
    ax_steps = ax_len.twinx()
    p1 = ax_steps.bar(df_plot['global_epoch'], df_plot['steps'], color='lightcoral', alpha=0.5, label='Steps per Epoch')
    p2, = ax_len.plot(df_plot['global_epoch'], df_plot['avg_episode_length'], color='dodgerblue', marker='.', label='Avg Ep Length')

    if (df_plot['interpolation_factor'] != -1).any():
        axs['curriculum'].plot(df_plot['global_epoch'], df_plot['interpolation_factor'], color='teal', label='Curriculum Factor')
        axs['curriculum'].set_title('Reward Curriculum Interpolation Factor')
        axs['curriculum'].set_ylabel('Factor (0=Start, 1=End)')
        axs['curriculum'].set_ylim(-0.05, 1.05)

    axs['targets'].plot(df_plot['global_epoch'], df_plot['avg_targets_reached'], color='indigo', label='Avg Targets Reached', marker='.', linestyle='-')
    axs['targets'].set_title('Average Targets Reached per Episode')
    axs['targets'].set_ylabel('Targets')

    ax_rew = axs['rewards']
    pos_data_raw = [df_plot[name].rolling(5, min_periods=1, center=True).mean() for name in POSITIVE_REWARD_KEYS if name in df_plot.columns]
    pos_labels_raw = [name for name in POSITIVE_REWARD_KEYS if name in df_plot.columns]
    neg_data_raw = [df_plot[name].rolling(5, min_periods=1, center=True).mean() for name in NEGATIVE_REWARD_KEYS if name in df_plot.columns]
    neg_labels_raw = [name for name in NEGATIVE_REWARD_KEYS if name in df_plot.columns]

    pos_data, pos_labels = (zip(*random.sample(list(zip(pos_data_raw, pos_labels_raw)), len(pos_data_raw)))) if pos_data_raw else ([], [])
    neg_data, neg_labels = (zip(*random.sample(list(zip(neg_data_raw, neg_labels_raw)), len(neg_data_raw)))) if neg_data_raw else ([], [])

    cmap = matplotlib.colormaps.get_cmap('tab20')
    color_list = list(cmap.colors)
    random.shuffle(color_list)
    total_colors_needed = len(pos_data) + len(neg_data)
    if len(color_list) < total_colors_needed:
        factor = (total_colors_needed // len(color_list)) + 1
        color_list = color_list * factor
    bonus_colors = color_list[:len(pos_data)]
    penalty_colors = color_list[len(pos_data):total_colors_needed]

    all_handles, all_labels = [], []
    if pos_data:
        pos_handles = ax_rew.stackplot(df_plot['global_epoch'], pos_data, labels=pos_labels, colors=bonus_colors, alpha=0.8)
        all_handles.extend(pos_handles)
        all_labels.extend(pos_labels)
    if neg_data:
        neg_handles = ax_rew.stackplot(df_plot['global_epoch'], neg_data, labels=neg_labels, colors=penalty_colors, alpha=0.8)
        all_handles.extend(neg_handles)
        all_labels.extend(neg_labels)

    ax_rew.set_title('Smoothed Reward Components (Bonuses vs. Penalties)')
    ax_rew.set_ylabel('Reward Value per Step')
    ax_rew.axhline(0, color='black', linewidth=0.75, linestyle='-')

    axs['return'].set_title('Average Episode Return')
    axs['return'].set_ylabel('Return')
    axs['loss'].set_title('Policy and Value Loss')
    axs['loss'].set_ylabel('Loss')
    axs['loss'].set_yscale('log')
    axs['kl_clip'].set_title('KL Divergence & Clip Fraction')
    axs['kl_clip'].set_ylabel('Value')
    ax_len.set_title('Episode Length & Steps per Epoch')
    ax_len.set_ylabel('Avg Ep Length', color='dodgerblue')
    ax_steps.set_ylabel('Total Steps', color='lightcoral')

    for key, ax in axs.items():
        ax.set_xlabel('Global Epoch')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        for start_epoch, end_epoch, _ in session_epoch_ranges:
            ax.axvline(x=start_epoch, color='cyan', linestyle=':', linewidth=1.2)
            ax.axvline(x=end_epoch, color='magenta', linestyle='--', linewidth=1.2)

    for key in ['return', 'loss', 'kl_clip', 'curriculum', 'targets']:
        if key in axs and (len(axs[key].get_lines()) > 0 or len(axs[key].patches) > 0):
            axs[key].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
    ax_len.legend(handles=[p1, p2], loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize='small')

    if all_handles:
        ax_rew.legend(handles=all_handles[::-1], labels=all_labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.25),
            ncol=6,
            fontsize='small', frameon=False)

    fig.subplots_adjust(left=0.06, right=0.85, top=0.90, bottom=0.2)
    fig.autofmt_xdate(rotation=30, ha='right')
    proxy_start = plt.Line2D([0], [0], color='cyan', linestyle=':', label='Session Start')
    proxy_end = plt.Line2D([0], [0], color='magenta', linestyle='--', label='Session End')
    fig.legend(handles=[proxy_start, proxy_end], loc='upper right')

    plt.show()


if __name__ == '__main__':
    plot_training_metrics()
