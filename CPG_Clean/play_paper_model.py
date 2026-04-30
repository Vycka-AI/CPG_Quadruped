import gymnasium as gym
import numpy as np
import os
import time
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from unitree_env_fixed import UnitreeEnv
from paper_imitation_env import PaperImitationEnvRelaxed

# --- CONFIGURATION ---
# Use 'paper_final.zip' or your latest checkpoint

MODEL_PATH = "./checkpoints_paper_refined/paper_refined_21000000_steps.zip" # Check exact name
MODEL_PATH = "./checkpoints_paper_refined/paper_refined_30000000_steps.zip" # Check exact name

#MODEL_PATH = "paper_final_fixed.zip" 
STATS_PATH = "paper_vecnormalize_fixed.pkl"
XML_PATH = '../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'

# Test Schedule: [Vx, Vy, Wz, Freq]
COMMAND_CYCLE = [
    ("SLOW WALK",   np.array([0.3, 0.0, 0.0, 2.5])),
    ("NORMAL TROT", np.array([0.6, 0.0, 0.0, 2.5])),
    ("FAST RUN",    np.array([1.0, 0.0, 0.0, 3.0])),
    ("TURBO DASH",  np.array([2.0, 0.0, 0.0, 4.0])), # <--- The Big Test
    ("BACKWARDS",   np.array([-0.4, 0.0, 0.0, 2.5])),
    ("SPIN LEFT",   np.array([0.0, 0.0, 1.0, 2.5])),
    ("STRAFE",      np.array([0.0, -0.4, 0.0, 2.5]),)
]
SWITCH_INTERVAL = 5.0 # Seconds per command

def make_env():
    # render_mode="human" opens the interactive window
    env = UnitreeEnv(model_path=XML_PATH, render_mode="human")
    # WRAPPER IS CRITICAL: Adds the Future Goals & History to observations
    env = PaperImitationEnvRelaxed(env)
    return env

def main():
    print(f"--- PLAYING PAPER IMPLEMENTATION MODEL ---")
    
    # 1. Setup Env
    env = DummyVecEnv([make_env])
    
    # 2. Load Stats (CRITICAL)
    if os.path.exists(STATS_PATH):
        print(f"Loading Stats: {STATS_PATH}")
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False 
        env.norm_reward = False
    else:
        print("!!! WARNING: Stats file not found. Robot will likely fail. !!!")

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"!!! Error: Model {MODEL_PATH} not found.")
        return

    model = PPO.load(MODEL_PATH)
    print("Model Loaded. Starting...")

    obs = env.reset()
    
    # Access internals for control
    wrapper = env.envs[0]       # PaperImitationEnv
    unitree = wrapper.env       # UnitreeEnv
    
    # Init Camera
    if unitree.viewer is not None:
        unitree.viewer.cam.distance = 3.0
        unitree.viewer.cam.elevation = -20
        unitree.viewer.cam.lookat[:] = unitree.data.qpos[:3]

    start_time = time.time()
    last_switch = start_time
    cmd_idx = 0
    
    # Set Initial Command
    label, cmd = COMMAND_CYCLE[0]
    wrapper.command = cmd
    print(f"Command: {label} {cmd}")

    try:
        while True:
            # 1. Predict Action
            # deterministic=True is safer for testing
            action, _ = model.predict(obs, deterministic=True)
            
            # 2. Step
            obs, _, dones, _ = env.step(action)
            
            # 3. Camera Tracking
            if unitree.viewer is not None:
                unitree.viewer.cam.lookat[:] = unitree.data.qpos[:3]
            
            # 4. Command Cycling
            now = time.time()
            if now - last_switch > SWITCH_INTERVAL:
                cmd_idx = (cmd_idx + 1) % len(COMMAND_CYCLE)
                label, cmd = COMMAND_CYCLE[cmd_idx]
                
                # Update the command in the wrapper
                wrapper.command = cmd
                print(f" >> SWITCH: {label} | Vel: {cmd[:3]} | Freq: {cmd[3]}")
                
                last_switch = now
            
            # Optional: Add small push to test robustness
            # if np.random.rand() < 0.005:
            #    unitree.data.xfrc_applied[1, 3:6] = [0, 50, 0] # 50N Shove

            time.sleep(0.005) # Cap framerate slightly

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
