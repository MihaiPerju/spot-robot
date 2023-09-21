import mujoco

import numpy as np
import wandb

from environment import SpotEnvironment

class SpotEnvironmentNaturalMotion(SpotEnvironment):
    def get_observation(self):
      bodies=[
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

      # reached_distance = self.data.body("trunk").xpos[0]
      # distance_to_goal = self.goal_distance - reached_distance

      trunk_rotation = self.get_z_axis_rotation()
    #   trunk_height = self.data.body("trunk").xpos[2]
    #   trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

      observation.append([
          trunk_rotation,
      ])

      for body in bodies:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        observation.append([
            *self.data.xpos[body_id],
            self.data.qpos[body_id],
            self.data.qvel[body_id],
        ])

        # is the foot in contact with the ground?
      feet_contact_obs=[]
      for site in ["FL","FR","RL","RR"]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        foot_location = self.data.site_xpos[site_id]

        foot_location_z = foot_location[2]
        # try:
        #     if foot_location_z<0.01:
        #         wandb.log({f"{site} Foot down":foot_location_z})
            
        #     wandb.log({f"{site} Foot time":self.data.time})
        # except Exception as e:
        #     pass

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

        self.n_steps +=1

        action=self.reshape(raw_action)
        self.data.ctrl=action

        update_time=self.data.time+self.action_delay
        while self.data.time<update_time:
            mujoco.mj_step(self.model, self.data)
            self.render()

        done=False
        if self.n_steps>=self.steps_per_episode:
            done=True
            self.n_steps=0

        reached_distance = self.data.body("trunk").xpos[0]
        progress_ratio = abs(reached_distance/self.goal_distance)

        z_axis_rotation=self.get_z_axis_rotation()
        trunk_height = self.data.body("trunk").xpos[2]
        trunk_orientation = self.get_y_axis_rotation()-0.69

        trunk_orientation_reward = -abs(trunk_orientation)*progress_ratio
        trunk_height_reward = (trunk_height-0.1)*progress_ratio

        # the default reward is based on how far it got compared to destination
        # the closer to the destination - the more the reward will increase
        reward = reached_distance + trunk_orientation_reward+trunk_height_reward

        wandb.log({"Trunk Height": trunk_height})
        wandb.log({"Trunk Height Reward": trunk_height_reward})
        wandb.log({"Trunk Orientation": trunk_orientation})
        wandb.log({"Trunk Orientation Reward": trunk_orientation_reward})
        wandb.log({"Reached distance": reached_distance})
        wandb.log({"Time": self.data.time})

        if reached_distance>self.max_distance:
          self.max_distance=reached_distance

        if reached_distance<self.last_distance:
          reward=-0.1
          
        self.last_distance=reached_distance

        if z_axis_rotation<0:
            reward=-1
            done=True
            self.n_steps=0

        # if the robot reached the destination - it won
        elif reached_distance>self.goal_distance:
            reward*=5
            done=True

        info={}
        observation = self.get_observation()

        return observation, reward, done, info
