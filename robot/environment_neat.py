import mujoco

import numpy as np
import wandb

import environment


class SpotEnvironment(environment.SpotEnvironment):
    def get_observation(self):
        bodies = [
            'trunk',

            'FL_calf',
            'FL_thigh',

            'FR_calf',
            'FR_thigh',

            'RL_calf',
            'RL_thigh',

            'RR_calf',
            'RR_thigh',
        ]
        observation = []

        trunk_rotation = self.get_z_axis_rotation()
        reached_distance = self.data.body("trunk").xpos[0]
        # distance_to_goal = self.goal_distance - reached_distance
        # trunk_height = self.data.body("trunk").xpos[2]
        # trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

        observation.append([
            trunk_rotation,
            reached_distance,
        ])

        for body in bodies:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            observation.append([
                self.data.qpos[body_id],
                self.data.qvel[body_id],
                # *self.data.xpos[body_id],
            ])

            # is the foot in contact with the ground?
        feet_contact_obs = []
        for site in ["FL", "FR", "RL", "RR"]:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site)
            foot_location = self.data.site_xpos[site_id]

            foot_location_z = foot_location[2]
            if foot_location_z < 0.01:
                foot_location = 0
            else:
                foot_location = 1

            feet_contact_obs.append(foot_location_z)

        observation.append(feet_contact_obs)

        return np.concatenate(observation)

    def get_y_axis_rotation(self):
        def quaternion_to_rotation_matrix(q):
            w, x, y, z = q
            return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])

        quaternion = self.data.body("trunk").xquat
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)

        # Extract the local y-axis (forward vector) from the rotation matrix
        local_y_axis = rotation_matrix[:, 1]
        return local_y_axis[1]

    def step(self, raw_action):
        wandb.log({"Step": self.n_steps})

        self.n_steps += 1

        # rms = np.sqrt(np.mean((self.last_action-raw_action)**2))
        self.last_action = raw_action
        # wandb.log({"RMS": rms})

        action = self.reshape(raw_action)
        self.data.ctrl = action

        update_time = self.data.time+self.action_delay
        while self.data.time < update_time:
            mujoco.mj_step(self.model, self.data)
            self.render()

        done = False
        if self.n_steps >= self.steps_per_episode:
            done = True
            self.n_steps = 0

        reached_distance = self.data.body("trunk").xpos[0]
        distance_to_goal = self.goal_distance - reached_distance

        wandb.log({"Reached distance": reached_distance})

        z_axis_rotation = self.get_z_axis_rotation()

        distance_progress = reached_distance-self.distance_reached_prev
        wandb.log({"Distance progress": distance_progress})

        reward = -distance_to_goal
        self.distance_reached_prev = reached_distance

        if reached_distance > self.max_distance:
            self.max_distance = reached_distance
            wandb.log({"Max distance": reached_distance})

        if z_axis_rotation < 0:
            done = True
            self.distance_reached_prev = 0
            self.n_steps = 0

        info = {}
        observation = self.get_observation()

        return observation, reward, done, info

    def reshape(self, raw_action):
        expanded_action = np.array([
            0, *raw_action[0:2],
            0, *raw_action[2:4],
            0, *raw_action[4:6],
            0, *raw_action[6:8],
        ])

        action = self.rescale_action(expanded_action)
        return action

    def rescale_action(self, raw_action):
        low_bound = np.array([0, -0.686, -2.3]*4)
        # high_bound = np.array([0, 4.501, -0.888]*4)
        # new range of motion to prevent unnatural movement
        high_bound = np.array([0, 1.3, -1.3]*4)
        action_range = high_bound - low_bound
        rescaled_action = low_bound + ((raw_action + 1.0) * 0.5) * action_range
        return rescaled_action
