import copy
import random
import numpy as np
import math

from calcs import clamp, normalize_angle_signed, ang, tanh
from config import *


class stick:
    def __init__(self, size):
        self.size = size
        self.points = np.zeros((2, 2), dtype=np.float32)
        self.vel = np.zeros((2, 2), dtype=np.float32)
        self.center = np.zeros(2, dtype=np.float32)
        self.angle = 0.0
        self.previous_angle = 0.0
        self.angular_velocity = 0.0
        self.linear_velocity_magnitude = 0.0
        self.previous_linear_velocity_magnitude = 0.0
        self.drag = 0.0

    def set_physical_state(self, pos, angle, linear_vel_avg=np.array([0., 0.]), angular_vel=0.0):
        self.center = np.array(pos, dtype=np.float32)
        self.angle = normalize_angle_signed(angle)
        self.previous_angle = normalize_angle_signed(angle - angular_vel * TIME_STEP_DELTA)
        self._reposition_points()
        self.vel[0] = linear_vel_avg
        self.vel[1] = linear_vel_avg
        self.angular_velocity = angular_vel
        self.linear_velocity_magnitude = np.linalg.norm(linear_vel_avg)

    def _reposition_points(self):
        half_size_x = self.size * math.cos(self.angle) / 2
        half_size_y = self.size * math.sin(self.angle) / 2
        self.points[0] = self.center - np.array([half_size_x, half_size_y])
        self.points[1] = self.center + np.array([half_size_x, half_size_y])

    def update(self, gravity_constant, delta_time_step):
        self.previous_linear_velocity_magnitude = self.linear_velocity_magnitude
        current_angle_for_angular_vel_calc = self.angle
        self.vel *= (1 - self.drag)
        self.vel[:, 1] += gravity_constant * delta_time_step
        self.points += self.vel * delta_time_step
        self.center = (self.points[0] + self.points[1]) / 2
        self.angle = normalize_angle_signed(ang(tuple(self.points[0]), tuple(self.points[1])))
        angle_diff = normalize_angle_signed(self.angle - current_angle_for_angular_vel_calc)
        self.angular_velocity = angle_diff / delta_time_step if delta_time_step > 0 else 0.0
        center_vel = (self.vel[0] + self.vel[1]) / 2
        self.linear_velocity_magnitude = np.linalg.norm(center_vel)
        self._reposition_points()


class Drone:
    def __init__(self, initial_pos, size, envDiag):
        self.starting_pos = initial_pos
        self.size = size
        self.envDiag = envDiag
        self.droneSpeedScalar = DRONE_MAX_SPEED_SCALAR
        self.droneRotationScalar = DRONE_MAX_ROT_SCALAR
        self.shell = stick(self.size)
        self.maxThrusterForce = MAX_THRUSTER_FORCE
        self.thruster_forces = [0.0, 0.0]
        self.previous_thruster_forces = [0.0, 0.0]
        self.thruster_force_delta_rate = THRUSTER_FORCE_DELTA_RATE
        self.currentObservationState = None

    def reset(self, initial_pos=None):
        actual_initial_pos = initial_pos if initial_pos is not None else self.starting_pos
        random_angle = random.uniform(-math.pi / 4, math.pi / 4)
        random_linear_vel_x = random.uniform(-INITIAL_MAX_LINEAR_VEL, INITIAL_MAX_LINEAR_VEL)
        random_linear_vel_y = random.uniform(-INITIAL_MAX_LINEAR_VEL, INITIAL_MAX_LINEAR_VEL)
        random_angular_vel = random.uniform(-INITIAL_MAX_ANGULAR_VEL, INITIAL_MAX_ANGULAR_VEL)
        self.shell.set_physical_state(actual_initial_pos, random_angle, np.array([random_linear_vel_x, random_linear_vel_y], dtype=np.float32), random_angular_vel)
        self.thruster_forces = [0.0, 0.0]
        self.previous_thruster_forces = [0.0, 0.0]
        self.currentObservationState = np.zeros(OBS_DIM)

    def get_observation(self, target_pos):
        center_vel = self.shell.vel.mean(axis=0)
        rel_x_pos = (target_pos[0] - self.shell.center[0]) / self.envDiag
        rel_y_pos = (target_pos[1] - self.shell.center[1]) / self.envDiag
        target_vector = np.array(target_pos) - self.shell.center
        target_dist = np.linalg.norm(target_vector)
        target_direction_normalized = target_vector / (target_dist + 1e-8)
        drone_velocity_vector = self.shell.vel.mean(axis=0)
        velocity_towards_target = np.dot(drone_velocity_vector, target_direction_normalized)

        drone_x, drone_y = self.shell.center
        h_dist_to_wall = min(drone_x, ENV_WIDTH - drone_x)
        v_dist_to_wall = min(drone_y, ENV_HEIGHT - drone_y)
        normalized_h_dist = h_dist_to_wall / (ENV_WIDTH / 2.0)
        normalized_v_dist = v_dist_to_wall / (ENV_HEIGHT / 2.0)

        # --- Time-to-Impact Calculation ---
        h_tti = 3.0  # Default to a safe, high value (e.g., 3 seconds)
        if center_vel[0] > 1.0:  # Moving right with some speed
            dist_to_right_wall = ENV_WIDTH - drone_x
            h_tti = dist_to_right_wall / center_vel[0]
        elif center_vel[0] < -1.0:  # Moving left with some speed
            dist_to_left_wall = drone_x
            h_tti = dist_to_left_wall / -center_vel[0]

        v_tti = 3.0  # Default to a safe, high value
        if center_vel[1] > 1.0:  # Moving down with some speed
            dist_to_bottom_wall = ENV_HEIGHT - drone_y
            v_tti = dist_to_bottom_wall / center_vel[1]
        elif center_vel[1] < -1.0:  # Moving up with some speed
            dist_to_top_wall = drone_y
            v_tti = dist_to_top_wall / -center_vel[1]

        # Use inverse TTI and tanh to create a bounded, normalized observation
        # This value approaches 1 as TTI approaches 0, and is 0 for large TTI
        normalized_h_tti = tanh(1.0 / (h_tti + 1e-6))
        normalized_v_tti = tanh(1.0 / (v_tti + 1e-6))

        observation = np.array(
            [tanh(center_vel[0] / self.droneSpeedScalar), tanh(center_vel[1] / self.droneSpeedScalar), tanh(self.shell.angular_velocity / self.droneRotationScalar), tanh(self.shell.linear_velocity_magnitude / (self.droneSpeedScalar * np.sqrt(2))), rel_x_pos, rel_y_pos, self.shell.angle, self.thruster_forces[0], self.thruster_forces[1], math.sin(self.shell.angle), math.cos(self.shell.angle), tanh(velocity_towards_target / self.droneSpeedScalar), normalized_h_dist, normalized_v_dist,
                normalized_h_tti, normalized_v_tti], dtype=np.float32)

        self.currentObservationState = observation
        return observation

    def apply_action(self, action, delta_time_step):
        self.previous_thruster_forces = copy.deepcopy(self.thruster_forces)
        self.thruster_forces[0] = clamp(0, 1, self.thruster_forces[0] + action[0] * self.thruster_force_delta_rate)
        self.thruster_forces[1] = clamp(0, 1, self.thruster_forces[1] + action[1] * self.thruster_force_delta_rate)

    def physics_move(self, gravity, delta_time_step):
        thruster_1_absolute_angle = normalize_angle_signed(THRUSTER_FIXED_RELATIVE_ANGLE + self.shell.angle)
        thruster_2_absolute_angle = normalize_angle_signed(THRUSTER_FIXED_RELATIVE_ANGLE + self.shell.angle)
        self.shell.vel[0][0] -= self.thruster_forces[0] * math.cos(thruster_1_absolute_angle) * delta_time_step * self.maxThrusterForce
        self.shell.vel[0][1] -= self.thruster_forces[0] * math.sin(thruster_1_absolute_angle) * delta_time_step * self.maxThrusterForce
        self.shell.vel[1][0] -= self.thruster_forces[1] * math.cos(thruster_2_absolute_angle) * delta_time_step * self.maxThrusterForce
        self.shell.vel[1][1] -= self.thruster_forces[1] * math.sin(thruster_2_absolute_angle) * delta_time_step * self.maxThrusterForce
        self.shell.update(gravity, delta_time_step)

    def draw(self, pygame_lib, screen, colMain, colSecondary, thruster_draw_size=5):
        p0_int = [int(self.shell.points[0][0]), int(self.shell.points[0][1])]
        p1_int = [int(self.shell.points[1][0]), int(self.shell.points[1][1])]

        pygame_lib.draw.line(screen, colMain, p0_int, p1_int, 2)

        pygame_lib.draw.circle(screen, colSecondary, p0_int, 3)
        pygame_lib.draw.circle(screen, colSecondary, p1_int, 3)

        thruster1_abs_draw_angle = normalize_angle_signed(THRUSTER_FIXED_RELATIVE_ANGLE + self.shell.angle)
        thruster1_end = [
            int(self.shell.points[0][0] + thruster_draw_size * math.cos(thruster1_abs_draw_angle)),
            int(self.shell.points[0][1] + thruster_draw_size * math.sin(thruster1_abs_draw_angle))
        ]
        pygame_lib.draw.line(screen, colSecondary, p0_int, thruster1_end, 2)

        thruster2_abs_draw_angle = normalize_angle_signed(THRUSTER_FIXED_RELATIVE_ANGLE + self.shell.angle)
        thruster2_end = [
            int(self.shell.points[1][0] + thruster_draw_size * math.cos(thruster2_abs_draw_angle)),
            int(self.shell.points[1][1] + thruster_draw_size * math.sin(thruster2_abs_draw_angle))
        ]
        pygame_lib.draw.line(screen, colSecondary, p1_int, thruster2_end, 2)