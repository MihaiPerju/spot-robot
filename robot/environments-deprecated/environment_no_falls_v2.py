import mujoco

import numpy as np
import wandb

from environment import SpotEnvironment

class SpotEnvironmentNoFalls(SpotEnvironment):
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
        wandb.log({"Reached distance": reached_distance})

        z_axis_rotation=self.get_z_axis_rotation()

        # the default reward is based on how far it got compared to destination
        # the closer to the destination - the more the reward will increase
        reward = reached_distance

        if reached_distance>self.max_distance:
            self.max_distance=reached_distance
        else:
            reward=-0.1

        if z_axis_rotation<0:
            wandb.log({ "Fall time":  self.data.time})
            reward=-1
            done=True
            self.n_steps=0

        # if the robot reached the destination - it won
        elif reached_distance>self.goal_distance:
            reward*=10
            done=True
        


        info={}
        observation = self.get_observation()
        wandb.log({ "Done": 0 if done==True else 1 })

        return observation, reward, done, info
