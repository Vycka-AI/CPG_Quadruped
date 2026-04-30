import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Corrected Import: Matches the file "CPG_Network.py" you have
from CPG_Network import CPG_Network_Paper

def analyze_trajectory():
    # 1. Setup
    dt = 0.01
    duration = 3.0 # seconds
    steps = int(duration / dt)
    
    # Initialize the specific class from your file
    cpg = CPG_Network_Paper(dt)
    
    # buffers for plotting
    time_log = []
    foot_pos_log = {
        'FL': {'x': [], 'y': [], 'z': []}, 
        'FR': {'x': [], 'y': [], 'z': []}
    }
    joint_log = []
    
    # Define Action: Forward Trot
    # Mu = 1.5 (Norm 0.0) -> Amplitude 
    # Omega = 2.25 Hz (Norm 0.0) -> Frequency
    # Psi = 0.0 (Norm 0.0) -> Straight
    action = np.zeros(12) 
    
    print(f"Simulating {duration} seconds of CPG trajectory...")

    for t in range(steps):
        # A. Step the CPG
        joints, obs = cpg.step(action)
        
        # B. Reconstruct Cartesian Foot Positions for Analysis
        # FIX: Access r and theta directly (The class uses polar coords internally)
        r = cpg.r
        theta = cpg.theta
        
        for i, name in enumerate(['FL', 'FR']): # Just plot front legs for clarity
            # --- RECONSTRUCTION OF CPG GEOMETRY ---
            step_scale = cpg.d_step * (r[i] - 1.0) # Note: (r - 1.0) is the scaling factor
            
            # 1. Raw Foot Position (Hip Frame)
            # Note: In the class, x is negative for forward motion standard
            # Psi is 0 here, so we simplify the cos(psi)/sin(psi) terms
            psi = 0.0
            raw_x = -cpg.d_step * (r[i] - 1.0) * np.cos(theta[i]) * np.cos(psi)
            
            x_f = raw_x 
            y_f = -cpg.d_step * (r[i] - 1.0) * np.cos(theta[i]) * np.sin(psi) # Should be 0.0 for straight
            
            # Add Hip Offset
            side_sign = 1 if i % 2 == 0 else -1
            y_f += (side_sign * cpg.l_hip)
            
            # Height Logic
            sin_theta = np.sin(theta[i])
            if sin_theta > 0:
                z_f = -cpg.h_nom + cpg.g_c * sin_theta
            else:
                z_f = -cpg.h_nom + cpg.g_p * sin_theta
                
            foot_pos_log[name]['x'].append(x_f)
            foot_pos_log[name]['y'].append(y_f)
            foot_pos_log[name]['z'].append(z_f)
            
        joint_log.append(joints[:3]) # Log FL joints
        time_log.append(t * dt)

    # --- PLOTTING ---
    print("Generating Plots...")
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("CPG Oscillator Foot Trajectory Analysis", fontsize=16)

    # 1. 3D Trajectory (FL Leg)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(foot_pos_log['FL']['x'], foot_pos_log['FL']['y'], foot_pos_log['FL']['z'], label='FL Foot')
    ax1.set_xlabel('X (Forward)')
    ax1.set_ylabel('Y (Side)')
    ax1.set_zlabel('Z (Height)')
    ax1.set_title('3D Foot Path (Hip Frame)')
    ax1.legend()

    # 2. Side View (X-Z) - Crucial for Step Height
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(foot_pos_log['FL']['x'], foot_pos_log['FL']['z'], 'b-', label='Swing/Stance')
    ax2.axhline(-cpg.h_nom, color='r', linestyle='--', label='Ground Level (h_nom)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (Gait Cycle)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()

    # 3. Top View (X-Y) - Crucial for "Spread" Issue
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(foot_pos_log['FL']['x'], foot_pos_log['FL']['y'], 'g-', label='FL')
    ax3.plot(foot_pos_log['FR']['x'], foot_pos_log['FR']['y'], 'm-', label='FR')
    
    # Draw Hip Markers
    ax3.scatter([0], [cpg.l_hip], color='g', marker='o', s=100, label='FL Hip')
    ax3.scatter([0], [-cpg.l_hip], color='m', marker='o', s=100, label='FR Hip')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Top View (Check for Spread)')
    ax3.grid(True)
    ax3.axis('equal')
    ax3.legend()

    # 4. Joint Angles (FL)
    ax4 = fig.add_subplot(2, 2, 4)
    joint_data = np.array(joint_log)
    ax4.plot(time_log, joint_data[:, 0], label='Hip Roll (Should be near 0)')
    ax4.plot(time_log, joint_data[:, 1], label='Hip Pitch')
    ax4.plot(time_log, joint_data[:, 2], label='Knee')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angle (rad)')
    ax4.set_title('FL Joint Commands')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_trajectory()