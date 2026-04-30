import numpy as np
import mujoco
import mujoco.viewer
import time
import sys

# Import your CPG class
from CPG_Network import CPG_Network_Paper

def test_inverse_kinematics():
    # --- 1. Setup Simulation ---
    # Update this path to your XML
    model_path = "../../../unitree_mujoco/unitree_robots/go2/scene_ground.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the 'model_path' variable in the script.")
        return

    # Initialize CPG (to get access to the IK function)
    dt = 0.01
    cpg = CPG_Network_Paper(dt)
    
    print("========================================")
    print("      INVERSE KINEMATICS STRESS TEST    ")
    print("========================================")
    print("1. FL Leg (Front Left): Draws a Vertical Circle")
    print("2. FR Leg (Front Right): Draws a Horizontal Line")
    print("3. Rear Legs: Stationary")
    print("\n[!] Watch for: Legs snapping/glitching (bad)")
    print("[!] Watch for: Smooth clamping at limits (good)")
    print("----------------------------------------")

    # Set robot to a neutral standing pose initially
    default_dof_pos = np.array([0.1, 0.8, -1.5] * 4)
    data.qpos[7:19] = default_dof_pos
    mujoco.mj_forward(model, data)

    # Launch Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            t = time.time() - start_time
            
            # --- GENERATE TEST TARGETS (In Hip Frame) ---
            
            # Leg 0 (FL): Vertical Circle (Test Height limits)
            # Center: 0.0m forward, -0.3m down
            # Radius: 0.15m (Should hit the safety limits at bottom)
            # Hip Offset (Approx natural stance width)
            # Left leg needs +0.1, Right leg needs -0.1
            y_natural_L = 0.1
            y_natural_R = -0.1

            # Leg 0 (FL): Vertical Circle (Now centered at natural width)
            x_fl = 0.1 * np.cos(2 * t)
            y_fl = y_natural_L + 0.05 * np.sin(2 * t) # <--- Added Offset
            z_fl = -0.30 + 0.15 * np.sin(2 * t) 

            # Leg 1 (FR): Vertical Circle (Mirrored)
            x_fr = 0.1 * np.cos(2 * t)
            y_fr = y_natural_R + 0.05 * np.sin(2 * t) # <--- Added Offset
            z_fr = -0.30 + 0.15 * np.sin(2 * t)
            
            # Rear Legs: Stay at default stand
            x_rear, y_rear, z_rear = 0.0, 0.0, -0.30

            # Construct Arrays for all 4 legs
            x_targets = np.array([x_fl, x_fr, x_rear, x_rear])
            y_targets = np.array([y_fl, y_fr, y_rear, y_rear])
            z_targets = np.array([z_fl, z_fr, z_rear, z_rear])

            # --- CALL INVERSE KINEMATICS ---
            try:
                # This calls the function you just fixed
                joint_targets = cpg._inverse_kinematics(x_targets, y_targets, z_targets)
            except Exception as e:
                print(f"CRITICAL MATH ERROR: {e}")
                break

            # --- SAFETY CHECK FOR NANS ---
            if np.isnan(joint_targets).any():
                print(f"\n[FAIL] NaN detected at t={t:.2f}!")
                print(f"Inputs FL -> X:{x_fl:.3f}, Y:{y_fl:.3f}, Z:{z_fl:.3f}")
                print("Your clamps are not working. Check sqrt() inputs.")
                break

            # --- APPLY TO ROBOT ---
            # We bypass the PD controller and set positions directly for this test
            # so we can see exactly what the math outputs.
            data.qpos[7:19] = joint_targets
            
            # Keep base fixed in air so we can see legs moving freely
            data.qpos[0:3] = [0, 0, 0.4] 
            data.qpos[3:7] = [1, 0, 0, 0] # Flat orientation
            data.qvel[:] = 0 # No physics velocity

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Slow down so human can see
            time.sleep(0.01)

if __name__ == "__main__":
    test_inverse_kinematics()