import numpy as np
import time
import mujoco
import mujoco.viewer
from unitree_env_fixed import UnitreeEnv
from CPG_Network_Enhanced import EnhancedHopfOscillator
from Phase_guided_logic import PhaseGuidedGenerator

def get_body_velocity(model, data):
    """ 
    Rotates the World Frame linear velocity into the Body Frame.
    Returns: Linear Velocity (Body), Angular Velocity (Body)
    """
    q = data.qpos[3:7]
    vel_world = data.qvel[:3]
    ang_vel_body = data.qvel[3:6] 
    
    # Quaternion to Rotation Matrix
    w, x, y, z = q
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x**2 + y**2)
    
    R = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])
    
    vel_body = R.T @ vel_world
    return vel_body, ang_vel_body

def print_statistics(name, samples, target):
    """Calculates and prints the mean/std deviation of velocity."""
    if not samples:
        return

    data = np.array(samples) # Shape: (N, 6) -> [Vx, Vy, Vz, Wx, Wy, Wz]
    
    # Calculate Mean and Std Dev
    mean_vel = np.mean(data, axis=0)
    std_vel = np.std(data, axis=0)
    
    print(f"\n" + "="*60)
    print(f"  RESULTS FOR COMMAND: {name}")
    print(f"  Target:  Vx={target[0]:.2f}, Vy={target[1]:.2f}, Wz={target[2]:.2f}")
    print(f"  AVERAGE: Vx={mean_vel[0]:.3f} (±{std_vel[0]:.3f}), Vy={mean_vel[1]:.3f} (±{std_vel[1]:.3f}), Wz={mean_vel[5]:.3f} (±{std_vel[5]:.3f})")
    print("="*60 + "\n")

def main():
    # Setup
    model_path = '../../unitree_mujoco/unitree_robots/go2/scene_ground.xml'
    env = UnitreeEnv(model_path=model_path, render_mode="human")
    
    obs, info = env.reset()

    # --- CAMERA SETUP ---
    if hasattr(env, 'viewer') and env.viewer is not None:
        env.viewer.cam.lookat[:] = env.data.qpos[:3]
        env.viewer.cam.distance = 3.0
        env.viewer.cam.elevation = -20
        env.viewer.cam.azimuth = 90
    
    motion_gen = PhaseGuidedGenerator(robot_type='go2')
    oscillators = [EnhancedHopfOscillator(dt=env.dt) for _ in range(4)]
    
    for i, osc in enumerate(oscillators):
        osc.reset()
        osc.phase_offset = 0.0 if i in [0, 3] else np.pi
        osc.omega = 2 * np.pi * 2.5 

    print("\n--- TEST: OMNIDIRECTIONAL TEACHER LOGIC (MEASUREMENT MODE) ---")

    start_time = time.time()
    
    # Measurement Logic Variables
    active_cmd_name = None
    cmd_start_time = 0.0
    velocity_samples = [] # Stores [vx, vy, vz, wx, wy, wz]
    SETTLING_TIME = 1.0   # Seconds to wait before measuring
    active_target = np.zeros(3)

    try:
        while True:
            t = time.time() - start_time
            cycle_t = t % 20.0 
            
            # 1. Determine Command
            cmd_name = "Hover"
            command = np.array([0.0, 0.0, 0.0])
            
            if 0 < cycle_t <= 4:
                cmd_name = "FORWARD"
                command = np.array([0.5, 0.0, 0.0])
            elif 4 < cycle_t <= 8:
                cmd_name = "STRAFE RIGHT"
                command = np.array([0.0, -0.3, 0.0])
            elif 8 < cycle_t <= 12:
                cmd_name = "TURN LEFT"
                command = np.array([0.0, 0.0, 0.8])
            elif 12 < cycle_t <= 16:
                cmd_name = "BACKWARD"
                command = np.array([-0.4, 0.0, 0.0])
            else:
                cmd_name = "STOP"
                command = np.array([0.0, 0.0, 0.0])

            # 2. Check for Command Switch
            if cmd_name != active_cmd_name:
                # If we were running a command, print its stats now
                if active_cmd_name is not None:
                    print_statistics(active_cmd_name, velocity_samples, active_target)
                
                # Reset for new command
                print(f"--> SWITCHING TO: {cmd_name}")
                active_cmd_name = cmd_name
                active_target = command
                cmd_start_time = t
                velocity_samples = []

            # 3. Step CPG
            phases = []
            for i, osc in enumerate(oscillators):
                osc.omega = 2 * np.pi * 2.5
                x, y = osc.step()
                phi = np.arctan2(y, x)
                phases.append(phi)
                
            # 4. Step IK
            foot_targets = motion_gen.get_foot_trajectory(phases, command)
            ref_joints = motion_gen.inverse_kinematics(foot_targets)
            env.step(ref_joints)

            # Camera Update
            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.cam.lookat[:] = env.data.qpos[:3]
            
            # 5. Measure Data (If settled)
            time_in_command = t - cmd_start_time
            
            if time_in_command > SETTLING_TIME:
                meas_lin, meas_ang = get_body_velocity(env.model, env.data)
                
                # Store full 6D velocity for analysis
                full_vel = np.concatenate([meas_lin, meas_ang])
                velocity_samples.append(full_vel)
                
                # Optional: Live print every 0.2s to show it's working
                if int(time_in_command * 100) % 20 == 0:
                     print(f"   Measuring... Vx: {meas_lin[0]:.2f} | Vy: {meas_lin[1]:.2f} | Wz: {meas_ang[2]:.2f}")

            # Manual sync
            time.sleep(env.dt)

    except KeyboardInterrupt:
        # Print stats for the final command if stopped manually
        print_statistics(active_cmd_name, velocity_samples, active_target)
        print("Stopped.")
        env.close()

if __name__ == "__main__":
    main()