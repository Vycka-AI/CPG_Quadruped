import numpy as np
import time
from unitree_env_paper import UnitreeEnv

def main():
    # --- CONFIG ---
    # Update path to your XML
    model_path = "../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml"
    
    print("Initializing Trot Controller...")
    env = UnitreeEnv(model_path=model_path, render_mode="human", frame_skip=10)
    
    obs, _ = env.reset()
    
    # --- DEFINING THE TROT GAIT ---
    # The CPG expects normalized actions in [-1, 1].
    # CPG Ranges:
    # mu (Amplitude):    [1.0, 2.0]  -> Middle (1.5) is 0.0
    # omega (Frequency): [0.0, 4.5]  -> Middle (2.25) is 0.0
    # psi (Steering):    [-1.5, 1.5] -> Middle (0.0) is 0.0
    
    # 1. Walk in Place (Amplitude=1.0)
    # Norm -1.0 maps to mu=1.0 (step height 0)
    action_stand = np.zeros(12)
    action_stand[0::3] = -1.0 # Set all mu to min (1.0)
    
    # 2. Forward Trot (Amplitude=1.5, Freq=2.25Hz)
    # All zeros maps to the middle of the ranges defined in CPG_Network.py
    action_trot = np.zeros(12) 
    
    print("1. Settling (Standing)...")
    for i in range(100):
        env.step(action_stand)
        time.sleep(0.01)

    print("2. STARTING TROT!")
    start_time = time.time()
    
    try:
        while True:
            # We just send the static trot command.
            # The CPG handles the rhythmic leg coordination internally.
            obs, reward, terminated, truncated, info = env.step(action_trot)
            
            if terminated:
                print("Robot fell! Resetting...")
                env.reset()
                
            # Optional: Add small sleep to not run super fast
            # (Mujoco viewer syncs usually, but this helps stability)
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("Stopping...")
        env.close()

if __name__ == "__main__":
    main()