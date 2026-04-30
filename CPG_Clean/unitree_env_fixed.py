import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import time

class UnitreeEnv(gym.Env):
    def __init__(self, model_path, render_mode=None, frame_skip=4):
        super().__init__()
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.nu = self.model.nu
        self.actuator_joint_ids = self.model.actuator_trnid[:, 0].astype(int)

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
        # Standard Go2 Standing Pose (Matches corrected IK target roughly)
        self.default_dof_pos = np.array([
            0.1, 0.8, -1.5,   # FL
            -0.1, 0.8, -1.5,  # FR
            0.1, 1.0, -1.5,   # RL
            -0.1, 1.0, -1.5   # RR
        ])
        
        # Strong PID for stability
        self.p_gains = np.full(self.nu, 80.0) 
        self.d_gains = np.full(self.nu, 2.0)
        self.dt = self.model.opt.timestep * self.frame_skip

        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Set Pose
        self.data.qpos[7:19] = self.default_dof_pos
        self.data.qpos[2] = 0.35
        
        # Noise
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, size=12)
        self.data.qvel[:] = np.random.uniform(-0.1, 0.1, size=self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        target_dof_pos = action
        
        for _ in range(self.frame_skip):
            current_pos = self.data.qpos[7:19]
            current_vel = self.data.qvel[6:18]
            
            torques = self.p_gains * (target_dof_pos - current_pos) + \
                      self.d_gains * (0 - current_vel)
            
            self.data.ctrl[:] = np.clip(torques, -25, 25)
            mujoco.mj_step(self.model, self.data)
            
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
            
        return self._get_obs(), 0.0, self.data.qpos[2] < 0.15, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()