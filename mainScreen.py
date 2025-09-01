import pygame
import math
import os
import torch
import numpy as np

from drone import Drone
from agent import PPOAgent
from calcs import distance
from text import drawText
from fontDict import fonts
from config import *
from fileManagement import list_existing_agents, find_latest_session_path, load_agent_config

pygame.init()

# --- Agent Selection and Loading ---
current_obs_dim = OBS_DIM
current_act_dim = ACT_DIM
current_hidden_dims = hiddenDims
selected_agent_name = None
agent_path = None
latest_weights_file = None
agents = list_existing_agents()
if not agents:
    print("No existing agents found. Please train an agent first using train.py.")
    pygame.quit()
    exit()
print("Existing agents:")
for i, agent in enumerate(agents):
    print(f"  [{i + 1}] {agent}")
while True:
    try:
        choice = int(input("Enter the number of the agent to visualize: "))
        if 1 <= choice <= len(agents):
            selected_agent_name = agents[choice - 1]
            agent_path = os.path.join("runs", selected_agent_name)
            break
        else:
            print("Invalid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")
try:
    loaded_config = load_agent_config(agent_path)
    current_obs_dim = loaded_config.get('OBS_DIM', OBS_DIM)
    current_act_dim = loaded_config.get('ACT_DIM', ACT_DIM)
    current_hidden_dims = tuple(loaded_config.get('hiddenDims', hiddenDims))
except FileNotFoundError:
    print(f"Config file not found for agent '{selected_agent_name}'. Using default config.")
except Exception as e:
    print(f"Error loading config for agent '{selected_agent_name}': {e}. Using default config.")
latest_session_path = find_latest_session_path(agent_path)
if latest_session_path:
    latest_weights_file = os.path.join(latest_session_path, "ppo_drone_agent_weights_latest.pkl")
    if not os.path.exists(latest_weights_file):
        print(f"WARNING: Weights file not found in latest session '{latest_session_path}'. Cannot load agent.")
        pygame.quit()
        exit()
else:
    print(f"No sessions found for agent '{selected_agent_name}'. Cannot load agent weights.")
    pygame.quit()
    exit()
print(f"Visualizing agent: {selected_agent_name} (OBS_DIM={current_obs_dim}, ACT_DIM={current_act_dim}, HiddenDims={current_hidden_dims})")
print(f"Loading weights from: {latest_weights_file}")
ObservationSpace = type('obs_space', (object,), {'shape': (current_obs_dim,)})()
ActionSpace = type('act_space', (object,), {'shape': (current_act_dim,), 'high': ACTION_HIGH, 'low': ACTION_LOW})()

# --- Mode Selection ---
game_mode = None
while game_mode not in ['mouse', 'path', 'm', 'p']:
    game_mode = input("Select mode: 'm' (mouse) or 'p' (path): ").lower()
if game_mode == 'm': game_mode = 'mouse'
if game_mode == 'p': game_mode = 'path'

# --- Screen and Asset Setup ---
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
clock = pygame.time.Clock()
FPS_RENDER = 60
scaleDownFactor = 1
screen_width = int(screen.get_width() / scaleDownFactor)
screen_height = int(screen.get_height() / scaleDownFactor)
env_offset_x = (screen_width - ENV_WIDTH) / 2
env_offset_y = (screen_height - ENV_HEIGHT) / 2
env_rect = pygame.Rect(env_offset_x, env_offset_y, ENV_WIDTH, ENV_HEIGHT)
drone_initial_spawn_pos_fixed = env_rect.center
screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()
montserratRegularAdaptive = fonts[f"regular{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratSmallAdaptive = fonts[f"regular{int(18 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratBoldAdaptive = fonts[f"bold{int(35 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratSmallBoldAdaptive = fonts[f"bold{int(18 / (scaleDownFactor ** (1 / 1.5)))}"]


class Endesga:
    black = [19, 19, 19];
    white = [255, 255, 255];
    greyL = [200, 200, 200];
    greyD = [100, 100, 100];
    greyVD = [50, 50, 50];
    very_light_blue = [199, 207, 221];
    my_blue = [32, 36, 46];
    debug_red = [255, 96, 141];
    network_green = [64, 128, 67];
    cream = [237, 171, 80]


# --- Game State and Agent Initialization ---
toggle = True
d = Drone(drone_initial_spawn_pos_fixed, DRONE_SIZE, ENV_DIAG)
agent_params_for_loading = {'observation_space': ObservationSpace, 'action_space': ActionSpace, 'ac_kwargs': {'hidden_sizes': current_hidden_dims}, 'seed': 0, 'steps_per_epoch': 1, 'epochs': 1, 'gamma': 0.99, 'clip_ratio': 0.2, 'pi_lr': 1e-3, 'vf_lr': 1e-3, 'train_pi_iters': 1, 'train_v_iters': 1, 'lam': 0.97, 'max_ep_len': 1, 'target_kl': 0.01}
loaded_agent = PPOAgent(**agent_params_for_loading)
loaded_agent.load_weights(latest_weights_file)
d.reset(initial_pos=drone_initial_spawn_pos_fixed)

# --- Path Mode Specific Setup ---
targets = [(env_rect.centerx, env_rect.top + 70), (env_rect.right - 70, env_rect.centery), (env_rect.centerx, env_rect.bottom - 70), (env_rect.left + 70, env_rect.centery), env_rect.center]
current_target_index = 0
TARGET_HIT_RADIUS = 25
paused_at_target = False
leg_start_time = pygame.time.get_ticks()
time_to_target = 0
arrival_speed = 0

if game_mode == 'path':
    target_pos = targets[current_target_index]
else:
    target_pos = env_rect.center

accumulated_time = 0.0
FIXED_PHYSICS_DELTA = TIME_STEP_DELTA

# --- Pre-calculate ring labels to handle overlaps ---
labels_by_radius = {}
# Collect proximity ring labels (numerical)
for radius in PROXIMITY_RADII:
    if radius not in labels_by_radius:
        labels_by_radius[radius] = []
    labels_by_radius[radius].append({'text': f"{radius:.0f}", 'color': Endesga.network_green, 'position': 'above'})
# Collect maneuver zone labels (text)
for name, radius in MANEUVER_THRESHOLDS.items():
    if 'ZONE' in name:
        if radius not in labels_by_radius:
            labels_by_radius[radius] = []
        labels_by_radius[radius].append({'text': name, 'color': Endesga.cream, 'position': 'below'})

pygame.mouse.set_visible(False)
running = True
while running:
    dt_real_seconds = clock.tick(FPS_RENDER) / 1000.0
    if not paused_at_target:
        accumulated_time += dt_real_seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                toggle = not toggle
            if event.key == pygame.K_r:
                d.reset(initial_pos=drone_initial_spawn_pos_fixed)
                if game_mode == 'path':
                    current_target_index = 0
                    target_pos = targets[current_target_index]
                    paused_at_target = False
                    leg_start_time = pygame.time.get_ticks()
            if event.key == pygame.K_RETURN and paused_at_target and game_mode == 'path':
                current_target_index = (current_target_index + 1) % len(targets)
                target_pos = targets[current_target_index]
                paused_at_target = False
                leg_start_time = pygame.time.get_ticks()

    if game_mode == 'mouse':
        mx, my = pygame.mouse.get_pos()
        target_pos = (mx / scaleDownFactor, my / scaleDownFactor)

    if not paused_at_target:
        while accumulated_time >= FIXED_PHYSICS_DELTA:
            obs = d.get_observation(target_pos)
            action = loaded_agent.actor_critic.act(torch.as_tensor(obs, dtype=torch.float32))
            d.apply_action(action, FIXED_PHYSICS_DELTA)
            d.physics_move(GRAVITY, FIXED_PHYSICS_DELTA)
            accumulated_time -= FIXED_PHYSICS_DELTA

    dist_to_current_target = distance(d.shell.center, target_pos)
    if game_mode == 'path' and not paused_at_target and dist_to_current_target < TARGET_HIT_RADIUS:
        paused_at_target = True
        time_to_target = (pygame.time.get_ticks() - leg_start_time) / 1000.0
        arrival_speed = d.shell.linear_velocity_magnitude
        d.shell.vel *= 0
        d.shell.angular_velocity = 0

    screen.fill(Endesga.my_blue)
    screen2.fill((0, 0, 0, 0))
    screenUI.fill((0, 0, 0, 0))
    pygame.draw.rect(screen2, Endesga.greyVD, env_rect, 2)

    # --- Draw rings and labels with top-right/bottom-right positioning ---
    for radius in sorted(labels_by_radius.keys(), reverse=True):
        labels_for_this_radius = labels_by_radius[radius]
        has_proximity = any(l['position'] == 'above' for l in labels_for_this_radius)
        has_maneuver = any(l['position'] == 'below' for l in labels_for_this_radius)
        if has_proximity:
            pygame.draw.circle(screen2, Endesga.network_green, (int(target_pos[0]), int(target_pos[1])), int(radius), 1)
        if has_maneuver:
            pygame.draw.circle(screen2, Endesga.cream, (int(target_pos[0]), int(target_pos[1])), int(radius), 1)
        above_labels = [l for l in labels_for_this_radius if l['position'] == 'above']
        below_labels = [l for l in labels_for_this_radius if l['position'] == 'below']
        line_height = 20
        padding = 5
        if above_labels:
            angle_above = -math.pi / 4
            base_x = int(target_pos[0] + radius * math.cos(angle_above) + padding)
            base_y = int(target_pos[1] + radius * math.sin(angle_above))
            for i, label_info in enumerate(above_labels):
                text_y = base_y - (i * line_height)
                drawText(screenUI, label_info['color'], montserratSmallBoldAdaptive, base_x, text_y, label_info['text'], Endesga.black, 1, justify='left')
        if below_labels:
            angle_below = math.pi / 4
            base_x = int(target_pos[0] + radius * math.cos(angle_below) + padding)
            base_y = int(target_pos[1] + radius * math.sin(angle_below))
            for i, label_info in enumerate(below_labels):
                text_y = base_y + (i * line_height)
                drawText(screenUI, label_info['color'], montserratSmallBoldAdaptive, base_x, text_y, label_info['text'], Endesga.black, 1, justify='left')

    if game_mode == 'path':
        for t_pos in targets:
            pygame.draw.circle(screen2, Endesga.greyD, (int(t_pos[0]), int(t_pos[1])), 8, 1)
        max_dist_for_color = ENV_DIAG / 2
        norm_dist = min(dist_to_current_target / max_dist_for_color, 1.0)
        line_color = (int(255 * norm_dist), int(255 * (1 - norm_dist)), 50)
        pygame.draw.line(screen2, line_color, d.shell.center, target_pos, 1)
        pygame.draw.circle(screen2, Endesga.network_green, (int(target_pos[0]), int(target_pos[1])), 12)
        pygame.draw.circle(screen2, Endesga.white, (int(target_pos[0]), int(target_pos[1])), 12, 1)
        d.draw(pygame, screen2, Endesga.very_light_blue, Endesga.greyL)
    else:
        pygame.draw.circle(screen2, Endesga.network_green, (int(target_pos[0]), int(target_pos[1])), 5, 5)
        d.draw(pygame, screen2, Endesga.white, Endesga.greyL)

    if toggle:
        items = {f"Agent: {selected_agent_name}": None, f"Session: {os.path.basename(os.path.dirname(latest_weights_file))}": None, "---": None, f"Render FPS: {round(clock.get_fps())}": None, f"Physics FPS: {PHYSICS_FPS}": None, f"Distance: {dist_to_current_target:.1f}": None}
        y_offset_factor = 0
        for label, prefix in reversed(items.items()):
            string = str(label)
            if prefix is not None: string = f"{prefix}: " + string
            drawText(screenUI, Endesga.debug_red, montserratRegularAdaptive, 5, screen_height - (30 + 25 * y_offset_factor), string, Endesga.black, 1, antiAliasing=False)
            y_offset_factor += 1

        # --- State value display logic ---
        drone_x, drone_y = d.shell.center
        center_vel = d.shell.vel.mean(axis=0)

        # Wall Distances
        h_dist_display = min(drone_x - env_rect.left, env_rect.right - drone_x) / (ENV_WIDTH / 2.0)
        v_dist_display = min(drone_y - env_rect.top, env_rect.bottom - drone_y) / (ENV_HEIGHT / 2.0)

        # Time-to-Impact (TTI)
        h_tti_display = float('inf')
        if center_vel[0] > 1.0:
            h_tti_display = (env_rect.right - drone_x) / center_vel[0]
        elif center_vel[0] < -1.0:
            h_tti_display = (drone_x - env_rect.left) / -center_vel[0]

        v_tti_display = float('inf')
        if center_vel[1] > 1.0:  # Pygame Y is inverted, so positive Y is down
            v_tti_display = (env_rect.bottom - drone_y) / center_vel[1]
        elif center_vel[1] < -1.0:
            v_tti_display = (drone_y - env_rect.top) / -center_vel[1]

        y_offset_factor = 0
        for i, state_item in enumerate(d.currentObservationState):
            label = abbreviatedStateNames[i] if i < len(abbreviatedStateNames) else f"OBS_{i}"

            # Default to the raw normalized value from the observation
            display_value = state_item

            if label == "hDistWall":
                display_value = h_dist_display
            elif label == "vDistWall":
                display_value = v_dist_display
            elif label == "h_TTI":
                display_value = h_tti_display
            elif label == "v_TTI":
                display_value = v_tti_display

            drawText(screenUI, Endesga.debug_red, montserratRegularAdaptive, screen_width - 250, screen_height - (30 + 25 * y_offset_factor), label, Endesga.black, 1, antiAliasing=False, justify='left')
            drawText(screenUI, Endesga.debug_red, montserratRegularAdaptive, screen_width - 5, screen_height - (30 + 25 * y_offset_factor), f"{display_value:.2f}", Endesga.black, 1, antiAliasing=False, justify='right')
            y_offset_factor += 1

        if paused_at_target and game_mode == 'path':
            paused_items = {"TARGET REACHED!": None, f"Time: {time_to_target:.2f} s": None, f"Arrival Speed: {arrival_speed:.2f} px/s": None, "Press ENTER to continue": None}
            y_offset_paused = 0
            for label, _ in paused_items.items():
                if label == "h_TTI":
                    label = "h_T2Impact"
                elif label == "v_TTI":
                    label = "v_T2Impact"
                drawText(screenUI, Endesga.cream, montserratBoldAdaptive, screen_width / 2, 50 + 45 * y_offset_paused, label, Endesga.black, 3, justify='center')
                y_offset_paused += 1

    screen.blit(pygame.transform.scale(screen2, screen.get_size()), (0, 0))
    screen.blit(pygame.transform.scale(screenUI, screen.get_size()), (0, 0))
    pygame.display.update()

pygame.quit()
