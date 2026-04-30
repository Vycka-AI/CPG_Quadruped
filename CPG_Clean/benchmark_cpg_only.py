import gymnasium as gym
import numpy as np
import time
import pandas as pd
from unitree_env_fixed import UnitreeEnv
from CPG_Network_Enhanced import EnhancedHopfOscillator
from Phase_guided_logic_new import PhaseGuidedGenerator

# --- CONFIG ---
XML_PATH = '../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'

TEST_CASES = [
    ("Slow Walk",       [0.3,  0.0,  0.0,  2.5]),
    ("Medium Trot",     [0.6,  0.0,  0.0,  2.5]),
    ("Fast Run",        [1.0,  0.0,  0.0,  3.0]),
    ("Turbo Dash",      [2.0,  0.0,  0.0,  4.0]), # The Hard One
    ("Ultra Dash",      [2.5,  0.0,  0.0,  4.0]), # The Hard One
    ("Side Dash",       [1.5,  0.0,  0.6,  3.0]), # The Hard One
    ("Side Dash",       [0.5,  0.0,  -0.6, 3.0]), # The Hard One
    ("Backward",        [-0.4, 0.0,  0.0,  2.5]),
    ("Strafe Right",    [0.0, -0.4,  0.0,  2.5]),
    ("Spin Left",       [0.0,  0.0,  1.0,  2.5]),
    ("Mixed Diagonal",  [0.5,  0.3,  0.0,  2.5]),
    ("Circle Walk",     [0.5,  0.0,  0.5,  2.5]),
]

SETTLING_TIME = 1.0
MEASURE_TIME = 2.0

def get_body_velocity(data):
    q = data.qpos[3:7] 
    vel_world = data.qvel[:3]
    ang_vel_body = data.qvel[3:6] 
    
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    return R.T @ vel_world, ang_vel_body

def main():
    print(f"--- BENCHMARKING PURE CPG (TEACHER) WITH AUTO-RESET ---")
    
    env = UnitreeEnv(model_path=XML_PATH, render_mode="human")
    env.reset()
    
    motion_gen = PhaseGuidedGenerator(robot_type='go2')
    oscillators = [EnhancedHopfOscillator(dt=env.dt) for _ in range(4)]
    
    results = []

    # Trot Offsets: FL(0), FR(pi), RL(pi), RR(0)
    phase_offsets = [0.0, np.pi, np.pi, 0.0] # Trot
    phase_offsets = [np.pi, 0.0, np.pi, 0.0] # Left right
    phase_offsets = [np.pi, np.pi, 0.0, 0.0] # Gallop

    try:
        for name, cmd_list in TEST_CASES:
            target_cmd = np.array(cmd_list) # [Vx, Vy, Wz, Freq]
            print(f"\nTesting: {name:15s} | Target: {target_cmd[:3]} | Freq: {target_cmd[3]}")
            
            # Reset CPG at start of new command to ensure clean slate
            for i, osc in enumerate(oscillators):
                osc.reset()
                osc.phase_offset = phase_offsets[i]
                osc.x = 1.0 
                osc.y = 0.0

            # --- Settling Phase ---
            start_settle = time.time()
            while time.time() - start_settle < SETTLING_TIME:
                phases = []
                freq = target_cmd[3]
                omega = 2 * np.pi * freq
                
                for osc in oscillators:
                    osc.omega = omega
                    x, y = osc.step()
                    phases.append(np.arctan2(y, x))
                
                foot_targets = motion_gen.get_foot_trajectory(phases, target_cmd[:3], frequency=freq)
                ref_joints = motion_gen.inverse_kinematics(foot_targets)
                
                # Step and Check Termination
                # UnitreeEnv returns: obs, reward, terminated, truncated, info
                _, _, terminated, _, _ = env.step(ref_joints)
                
                if terminated:
                    # ROBOT FELL! Reset physics AND CPG
                    env.reset()
                    for i, osc in enumerate(oscillators):
                        osc.reset()
                        osc.phase_offset = phase_offsets[i]
                        osc.x, osc.y = 1.0, 0.0
                
                if env.viewer is not None:
                    env.viewer.cam.lookat[:] = env.data.qpos[:3]
                time.sleep(0.007)

            # --- Measurement Phase ---
            measured_lin = []
            measured_ang = []
            
            start_measure = time.time()
            while time.time() - start_measure < MEASURE_TIME:
                phases = []
                freq = target_cmd[3]
                omega = 2 * np.pi * freq
                for osc in oscillators:
                    osc.omega = omega
                    x, y = osc.step()
                    phases.append(np.arctan2(y, x))

                foot_targets = motion_gen.get_foot_trajectory(phases, target_cmd[:3], frequency=freq)
                ref_joints = motion_gen.inverse_kinematics(foot_targets)
                
                # Step and Check Termination
                _, _, terminated, _, _ = env.step(ref_joints)
                
                if terminated:
                    env.reset()
                    for i, osc in enumerate(oscillators):
                        osc.reset()
                        osc.phase_offset = phase_offsets[i]
                        osc.x, osc.y = 1.0, 0.0

                lin, ang = get_body_velocity(env.data)
                measured_lin.append(lin)
                measured_ang.append(ang)
                
                if env.viewer is not None:
                    env.viewer.cam.lookat[:] = env.data.qpos[:3]
                time.sleep(0.007)

            # Stats
            avg_lin = np.mean(measured_lin, axis=0)
            avg_ang = np.mean(measured_ang, axis=0)
            
            vx_err = avg_lin[0] - target_cmd[0]
            vy_err = avg_lin[1] - target_cmd[1]
            wz_err = avg_ang[2] - target_cmd[2]
            
            results.append({
                "Test Name": name,
                "Tgt Vx": target_cmd[0], "Act Vx": avg_lin[0], "Err Vx": vx_err,
                "Tgt Vy": target_cmd[1], "Act Vy": avg_lin[1], "Err Vy": vy_err,
                "Tgt Wz": target_cmd[2], "Act Wz": avg_ang[2], "Err Wz": wz_err
            })

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()

    # Report
    print("\n" + "="*80)
    print(f"{'PURE CPG TEACHER BENCHMARK (FIXED & RESET)':^80}")
    print("="*80)
    df = pd.DataFrame(results)
    print(df[["Test Name", "Tgt Vx", "Act Vx", "Err Vx", "Tgt Vy", "Act Vy", "Err Vy"]].round(3).to_string(index=False))
    
    print("\n" + "-"*80)
    # Calculate Overall MSE
    mse_vx = np.mean(df["Err Vx"]**2)
    mse_vy = np.mean(df["Err Vy"]**2)
    print(f"Overall MSE (Vx): {mse_vx:.4f}")
    print(f"Overall MSE (Vy): {mse_vy:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()