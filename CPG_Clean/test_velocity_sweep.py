import numpy as np
import time
import mujoco
import mujoco.viewer
from unitree_env_fixed import UnitreeEnv
from CPG_Network_Enhanced import EnhancedHopfOscillator
from Phase_guided_logic_new import PhaseGuidedGenerator

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

def print_statistics(name, samples, target):
    if not samples: return
    data = np.array(samples)
    mean = np.mean(data, axis=0)
    
    if abs(target[0]) > 0:   # Vx
        err = abs(mean[0] - target[0])
        res_str = f"Vx={mean[0]:.3f}"
    elif abs(target[1]) > 0: # Vy
        err = abs(mean[1] - target[1])
        res_str = f"Vy={mean[1]:.3f}"
    elif abs(target[2]) > 0: # Wz
        err = abs(mean[5] - target[2])
        res_str = f"Wz={mean[5]:.3f}"
    else: # Zero
        err = np.linalg.norm(mean[:3])
        res_str = f"VelMag={err:.3f}"

    status = "[PASS]" if err < 0.05 else "[FAIL]"
    if name == "STOP": status = "[PASS]" if err < 0.02 else "[FAIL]"
    # Relax turn tolerance slightly as angular velocity is noisy
    if "TURN" in name: status = "[PASS]" if err < 0.08 else "[FAIL]"

    print(f"{status} {name:<15} | Tgt: {target} | Act: {res_str}")

def main():
    env = UnitreeEnv(model_path='../../unitree_mujoco/unitree_robots/go2/scene_ground.xml', render_mode="human")
    env.reset()
    
    if hasattr(env, 'viewer') and env.viewer is not None:
        env.viewer.cam.lookat[:] = env.data.qpos[:3]
        env.viewer.cam.distance = 3.5
        env.viewer.cam.azimuth = 90
    
    motion_gen = PhaseGuidedGenerator(robot_type='go2')
    oscillators = [EnhancedHopfOscillator(dt=env.dt) for _ in range(4)]
    
    for i, osc in enumerate(oscillators):
        osc.reset()
        osc.phase_offset = 0.0 if i in [0, 3] else np.pi

    print("\n--- OMNIDIRECTIONAL PERFORMANCE SWEEP (FIXED STOP) ---")
    print(f"{'STATUS':<6} {'TEST NAME':<15} | {'TARGET [Vx, Vy, Wz, Hz]':<25} | {'RESULT'}")
    print("-" * 70)

    start_time = time.time()
    
    schedule = [
        # --- DRIFT CHECK ---
        (3.0, "STOP",        0.0,  0.0,  0.0, 0.0), 
        
        # --- FORWARD ---
        (4.0, "FWD SLOW",    0.2,  0.0,  0.0, 2.5),
        (4.0, "FWD MED",     0.5,  0.0,  0.0, 2.5),
        (4.0, "FWD FAST",    0.8,  0.0,  0.0, 2.5),
        (4.0, "FWD HI-FREQ", 0.5,  0.0,  0.0, 3.5),

        # --- STRAFE ---
        (4.0, "STRAFE SLOW", 0.0, -0.2,  0.0, 2.5),
        (4.0, "STRAFE FAST", 0.0,  0.4,  0.0, 2.5),
        
        # --- TURN ---
        (4.0, "TURN SLOW",   0.0,  0.0,  0.5, 2.5),
        (4.0, "TURN FAST",   0.0,  0.0,  1.0, 2.5),
        
        # --- FINAL STOP ---
        (2.0, "STOP FINAL",  0.0,  0.0,  0.0, 0.0),
    ]

    current_stage = 0
    stage_start_time = time.time()
    velocity_samples = []
    
    try:
        while True:
            t = time.time() - start_time
            stage_t = time.time() - stage_start_time
            
            dur, name, vx, vy, wz, freq = schedule[current_stage]
            
            if stage_t > dur:
                print_statistics(name, velocity_samples, [vx, vy, wz, freq])
                
                current_stage += 1
                if current_stage >= len(schedule): break
                
                stage_start_time = time.time()
                velocity_samples = []
                continue

            # CPG
            omega = 2 * np.pi * freq
            phases = []
            for osc in oscillators:
                osc.omega = omega
                x, y = osc.step()
                phases.append(np.arctan2(y, x))
            
            # IK Logic
            cmd = np.array([vx, vy, wz])
            
            # Explicitly force standing if Frequency is 0 (STOP Mode)
            force_stand = (freq == 0.0)
            
            foot_targets = motion_gen.get_foot_trajectory(phases, cmd, frequency=freq, is_standing=force_stand)
            env.step(motion_gen.inverse_kinematics(foot_targets))

            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.cam.lookat[:] = env.data.qpos[:3]

            if stage_t > 1.0:
                lin, ang = get_body_velocity(env.data)
                velocity_samples.append(np.concatenate([lin, ang]))

            time.sleep(env.dt)

    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    main()