import numpy as np

class EnhancedHopfOscillator:
    def __init__(self, dt, alpha=50.0):
        self.dt = dt
        self.alpha = alpha
        self.x, self.y = 1.0, 0.0
        self.mu = 1.0
        self.omega = 2 * np.pi * 1.0
        self.phase_offset = 0.0
        
    def set_parameters(self, amplitude, frequency, phase_offset):
        # In this implementation, amplitude controls the radius of the limit cycle
        self.mu = amplitude**2 
        self.omega = 2 * np.pi * frequency
        self.phase_offset = phase_offset

    def _dynamics(self, x, y):
        r_sq = x**2 + y**2
        x_dot = self.alpha * (self.mu - r_sq) * x - self.omega * y
        y_dot = self.alpha * (self.mu - r_sq) * y + self.omega * x
        return x_dot, y_dot

    def step(self):
        # RK4 Integration
        k1x, k1y = self._dynamics(self.x, self.y)
        k2x, k2y = self._dynamics(self.x + 0.5 * self.dt * k1x, self.y + 0.5 * self.dt * k1y)
        k3x, k3y = self._dynamics(self.x + 0.5 * self.dt * k2x, self.y + 0.5 * self.dt * k2y)
        k4x, k4y = self._dynamics(self.x + self.dt * k3x, self.y + self.dt * k3y)

        self.x += (self.dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        self.y += (self.dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

        # Apply phase modulation for output
        # x is analogous to cos(theta), y is analogous to sin(theta)
        px = self.x * np.cos(self.phase_offset) - self.y * np.sin(self.phase_offset)
        py = self.x * np.sin(self.phase_offset) + self.y * np.cos(self.phase_offset)
        return px, py

class CPG_Network_Hopf_IK:
    def __init__(self, dt, half_circle = True):
        self.dt = dt
        self.oscillators = [EnhancedHopfOscillator(dt) for _ in range(4)]
        
        # --- Robot Geometry (Unitree Go2 approx) copied from Paper ---
        self.l_hip = 0.0955   
        self.l_thigh = 0.213  
        self.l_calf = 0.213   
        self.h_nom = 0.30 
        
        # --- Gait Parameters ---
        #self.step_length_scale = 0.05  # 10cm radius (20cm stride)
        #self.ground_clearance = 0.15   # 10cm height


        self.step_length_scale = 0.05   # 10cm radius -> 20cm total stride
        self.ground_clearance = 0.04    # 5cm lift (Standard)

        self.penetration_depth = 0.00  # How deep we push into ground during stance (optional)
        self.half_circle = half_circle
        # Range Configuration
        self.param_ranges = {
            'amplitude': (0.5, 1.5), # 1.0 is standard circle
            'frequency': (1.0, 4.0),
            'phase': (-np.pi, np.pi)
        }

    def step(self, rl_action):
        if self.half_circle:
            # 1. Parse Actions 
            # (Same as before...)
            action_matrix = rl_action.reshape(4, 3)
            amps = 0.0 + (action_matrix[:, 0] + 1)/2 * (1.5) 
            freqs = self.param_ranges['frequency'][0] + (action_matrix[:, 1] + 1)/2 * (self.param_ranges['frequency'][1] - self.param_ranges['frequency'][0])
            phases = action_matrix[:, 2] * np.pi 

            # Arrays for IK targets
            x_foot_global = np.zeros(4)
            y_foot_global = np.zeros(4)
            z_foot_global = np.zeros(4)

            cpg_states = []

            for i, osc in enumerate(self.oscillators):
                osc.set_parameters(amps[i], freqs[i], phases[i])
                x_osc, y_osc = osc.step() 
                cpg_states.extend([x_osc, y_osc])
                
                # --- TRAJECTORY SHAPING (SEMI-OVAL) ---
                
                # 1. Step Length (X-Axis of the Oval)
                # x_osc goes from -1 to 1. 
                # We scale it by 'step_length_scale' (e.g., 0.15m)
                # Negative sign ensures positive oscillator X = Foot Backward (Propulsion)
                x_foot_global[i] = -self.step_length_scale * x_osc

                # 2. Lateral Offset (Y-Axis)
                # Keeps legs apart so they don't collide
                side_sign = 1 if i % 2 == 0 else -1
                y_foot_global[i] = side_sign * self.l_hip

                # 3. Step Height (Z-Axis of the Oval)
                # We want a semi-oval: Curved on top, flat on bottom.
                
                if y_osc > 0:
                    # SWING PHASE (Upper half of the oval)
                    # y_osc (0 to 1) maps to Height (0 to ground_clearance)
                    z_foot_global[i] = -self.h_nom + (self.ground_clearance * y_osc)
                else:
                    # STANCE PHASE (Flat bottom)
                    # Force Z to be exactly at nominal height (or slightly lower for traction)
                    # We use a tiny bit of 'penetration' to ensure the sim physics detects contact
                    z_foot_global[i] = -self.h_nom - 0.005 # Push 5mm into ground

            # 4. Inverse Kinematics
            target_joint_pos = self._inverse_kinematics(x_foot_global, y_foot_global, z_foot_global)

            return target_joint_pos, np.array(cpg_states)
        else:
            # 1. Parse Actions (4 legs x 3 params)
            action_matrix = rl_action.reshape(4, 3)
            
            # Map actions to physical parameters
            # Amplitude > 0 required for movement
            amps = 0.0 + (action_matrix[:, 0] + 1)/2 * (1.5) 
            freqs = self.param_ranges['frequency'][0] + (action_matrix[:, 1] + 1)/2 * (self.param_ranges['frequency'][1] - self.param_ranges['frequency'][0])
            phases = action_matrix[:, 2] * np.pi 

            # Arrays to hold foot positions
            x_foot_global = np.zeros(4)
            y_foot_global = np.zeros(4)
            z_foot_global = np.zeros(4)

            cpg_states = []

            for i, osc in enumerate(self.oscillators):
                osc.set_parameters(amps[i], freqs[i], phases[i])
                
                # Hopf output: x_osc approx cos(t), y_osc approx sin(t)
                x_osc, y_osc = osc.step() 
                cpg_states.extend([x_osc, y_osc])
                
                # --- MAPPING HOPF TO FOOT TRAJECTORY ---
                # 1. Longitudinal (X): Driven by Oscillator X (Cosine-like)
                # When x_osc is positive, foot is forward.
                x_foot_global[i] = -self.step_length_scale * x_osc

                # 2. Lateral (Y): Fixed offset + slight yaw correction if needed (ignored for now)
                side_sign = 1 if i % 2 == 0 else -1
                y_foot_global[i] = side_sign * self.l_hip

                # 3. Vertical (Z): Driven by Oscillator Y (Sine-like)
                # If y_osc > 0 (Swing phase): Lift leg
                # If y_osc < 0 (Stance phase): Push slightly or stay flat
                if y_osc > 0:
                    z_foot_global[i] = -self.h_nom + self.ground_clearance * y_osc
                else:
                    z_foot_global[i] = -self.h_nom + self.penetration_depth * y_osc

            # 4. Inverse Kinematics (Joint Space conversion)
            target_joint_pos = self._inverse_kinematics(x_foot_global, y_foot_global, z_foot_global)

            return target_joint_pos, np.array(cpg_states)

    def _inverse_kinematics(self, x, y, z):
        """
        Copied and adapted from CPG_Network.py 
        Calculates joint angles q1, q2, q3 given foot coordinates.
        """
        joints = []
        epsilon = 1e-4 
        
        for i in range(4):
            side_sign = 1 if i % 2 == 0 else -1
            
            # --- 1. Hip Roll (q1) ---
            lyz = np.sqrt(y[i]**2 + z[i]**2)
            lyz = np.maximum(lyz, self.l_hip + epsilon)
            
            term_q1 = np.clip(side_sign * self.l_hip / lyz, -1.0, 1.0)
            q1 = np.arctan2(y[i], -z[i]) - np.arcsin(term_q1)
            
            # --- 2. Leg Virtual Length ---
            d_eff = np.sqrt(np.maximum(lyz**2 - self.l_hip**2, 0.0))
            x_eff = x[i]
            l_virtual = np.sqrt(x_eff**2 + d_eff**2)
            
            # Safety clamping (from your provided fix)
            max_reach = 0.38  
            min_reach = 0.15 
            l_virtual = np.clip(l_virtual, min_reach, max_reach)
            
            # --- 3. Knee Pitch (q3) ---
            cos_q3 = (l_virtual**2 - self.l_thigh**2 - self.l_calf**2) / (2 * self.l_thigh * self.l_calf)
            cos_q3 = np.clip(cos_q3, -1.0, 1.0)
            q3 = -np.arccos(cos_q3) 
            
            # --- 4. Hip Pitch (q2) ---
            alpha = np.arctan2(-x_eff, d_eff) 
            cos_beta = (l_virtual**2 + self.l_thigh**2 - self.l_calf**2) / (2 * l_virtual * self.l_thigh)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = np.arccos(cos_beta)
            q2 = alpha + beta
            
            joints.extend([q1, q2, q3])
            
        return np.array(joints)
    
    def reset(self):
        for osc in self.oscillators:
            osc.x, osc.y = 1.0, 0.0