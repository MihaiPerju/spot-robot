
import mujoco
import gym
from gym.utils import seeding
import numpy as np
import wandb
import cv2
import time


def get_standing_spot_model():
  # opening the scene
  xml_file_path = './scene.xml'
  with open(xml_file_path, 'r') as file:
    xml = file.read()

  spot = mujoco.MjModel.from_xml_string(xml)
  spot_data = mujoco.MjData(spot)
  keyframe = spot.keyframe("home")

  spot_data.qpos=keyframe.qpos
  spot_data.ctrl = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8,]

  return spot, spot_data


# ENVIRONMENT
class Recorder():
  def __init__(self, model, framerate=30, ):
    self.frames=[]
    self.framerate=framerate
    self.renderer = mujoco.Renderer(model, 780, 1280)

  def add_frame(self, model, data):
    model.cam_pos[0][0]=data.body("trunk").xpos[0]
    model.cam_pos[0][1]=data.body("trunk").xpos[1]-1.5

    self.renderer.update_scene(data, camera=0)
    frame = self.renderer.render()
    self.frames.append(frame)

  def show_video(self):
    # Determine width and height from the first frame
    h, w, layers = self.frames[0].shape
    size = (w, h)

    timestamp = int(time.time())
    output_path = f"videos/{timestamp}.avi"

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), self.framerate, size)

    # Write each frame to the video
    for frame in self.frames:
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      out.write(rgb_frame)

    out.release()
    wandb.log({"Timestamp": timestamp})

  def reset(self):
    self.frames=[]


class SpotEnvironment(gym.Env):
    def __init__(self,  steps_per_episode, goal_distance, should_render=False):
        self.model, self.data = get_standing_spot_model()
        self.action_delay=0.2
        self.should_render=should_render
        self.goal_distance=goal_distance
        self.max_distance=-1
        self.last_distance=-1

        self.n_steps=0
        self.steps_per_episode=steps_per_episode

        if should_render:
          self.recorder = Recorder(self.model)

        # Set random seed
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model, self.data = get_standing_spot_model()

        if self.should_render:
          self.recorder.reset()

        observation = self.get_observation()
        return observation

    def get_z_axis_rotation(self):

      def quaternion_to_rotation_matrix(q):
          w, x, y, z = q
          return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                          [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                          [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])

      quaternion = self.data.body("trunk").xquat
      rotation_matrix = quaternion_to_rotation_matrix(quaternion)

      # Step 2: Extract the local z-axis (up vector) from the rotation matrix
      local_z_axis = rotation_matrix[:, 2]

      # Step 3: Check if the object is upside down (opposite direction of the world's z-axis)
      result = local_z_axis[2] < 0
      # print(f"Z axis {local_z_axis[2]}")

      return local_z_axis[2]

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
      trunk_height = self.data.body("trunk").xpos[2]
      trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

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

      # for site in ["FL","FR","RL","RR"]:
      #   site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
      #   foot_location = self.data.site_xpos[site_id]
      #   observation.append(foot_location)

      return np.concatenate(observation)

    def rescale_action(self, raw_action):
      low_bound = np.array([0, -0.686, -2.818]*4)
      # high_bound = np.array([0, 4.501, -0.888]*4)
      high_bound = np.array([0, 2, -0.888]*4) # new range of motion to prevent unnatural movement

      action_range = high_bound - low_bound
      rescaled_action = low_bound + ((raw_action + 1.0) * 0.5) * action_range

      return rescaled_action

    def reshape(self, raw_action):
      expanded_action = np.array([
          0, *raw_action[[0,1]],
          0, *raw_action[[2,3]],
          0, *raw_action[[4,5]],
          0, *raw_action[[6,7]],
      ])

      action = self.rescale_action(expanded_action)
      return action

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
        reward = reached_distance/self.goal_distance

        # if the robot reached the destination - it won
        if reached_distance>self.goal_distance:
          reward*=10
          done=True

        elif reached_distance<0:
          reward*=1.5
        elif z_axis_rotation<0:
            reward*=2
        else:
            reward*=7

        info={}
        observation = self.get_observation()
        wandb.log({ "Done": 1 if done==True else 0 })

        return observation, reward, done, info

    def render(self):
      if not self.should_render:
        return

      if len(self.recorder.frames) < self.data.time * self.recorder.framerate:
        self.recorder.add_frame(self.model, self.data)

