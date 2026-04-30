import numpy as np

class CPG_Network_Paper:
    def __init__(self, dt, leg_params=None):
        self.dt = dt
        
        # --- Robot Geometry (Unitree Go2 approx) ---
        self.l_hip = 0.0955   
        self.l_thigh = 0.213  
        self.l_calf = 0.213   
        
        # Default standing height
        self.h_nom = 0.30 
        
        # --- Internal States ---
        self.r = np.ones(4)      
        self.r_dot = np.zeros(4)
        self.theta = np.zeros(4) 
        
        # Initialize phases (Trot Gait)
        self.theta = np.array([0.0, np.pi, np.pi, 0.0])

        # --- Constants from Paper (Tuned for Stability) ---
        self.a = 150.0  
        self.d_step = 0.15 
        
        # --- FIX 1: LOWER SWING HEIGHT ---
        # 0.10 was too high (causing self-collision/clamping)
        # 0.05 is standard for robust trotting
        self.g_c = 0.05    
        
        self.g_p = 0.0     

        # Action Ranges
        self.mu_range = [1.0, 2.0]       
        self.omega_range = [0.0, 4.5]    
        #self.psi_range = [-1.5, 1.5]     
        self.psi_range = [-np.pi, np.pi] #For backward

    def step(self, rl_action):
        # 1. Parse Actions
        action_matrix = rl_action.reshape(4, 3)
        mu = self._map(action_matrix[:, 0], *self.mu_range)
        omega = self._map(action_matrix[:, 1], *self.omega_range)
        psi = self._map(action_matrix[:, 2], *self.psi_range)

        # 2. Oscillator Dynamics (SUB-STEPPED FOR STABILITY)
        # We divide the large dt into smaller chunks to prevent Euler explosion
        integration_steps = 10 
        dt_segment = self.dt / integration_steps
        
        for _ in range(integration_steps):
            r_ddot = self.a * ((self.a / 4.0) * (mu - self.r) - self.r_dot)
            self.r_dot += r_ddot * dt_segment
            self.r += self.r_dot * dt_segment
            
            self.theta += (omega * 2 * np.pi) * dt_segment
            
        self.theta = self.theta % (2 * np.pi)

        # 3. Geometric Mapping to Foot Positions
        # Calculate deviations from the "Neutral" point
        x_foot = -self.d_step * (self.r - 1.0) * np.cos(self.theta) * np.cos(psi)
        y_foot_dev = -self.d_step * (self.r - 1.0) * np.cos(self.theta) * np.sin(psi)
        
        y_foot = np.zeros(4)
        z_foot = np.zeros(4)

        for i in range(4):
            # --- FIX 2: PERMANENT HIP OFFSET ---
            # Automatically add hip width so legs stand straight (not knock-kneed)
            side_sign = 1 if i % 2 == 0 else -1
            y_foot[i] = y_foot_dev[i] + (side_sign * self.l_hip)

            # Z Calculation
            sin_theta = np.sin(self.theta[i])
            if sin_theta > 0:
                z_foot[i] = -self.h_nom + self.g_c * sin_theta
            else:
                z_foot[i] = -self.h_nom + self.g_p * sin_theta

        # 4. Inverse Kinematics
        target_joint_pos = self._inverse_kinematics(x_foot, y_foot, z_foot)
        
        cpg_obs = np.concatenate([self.r, self.r_dot, np.cos(self.theta), np.sin(self.theta)])
        return target_joint_pos, cpg_obs

    def _inverse_kinematics(self, x, y, z):
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
            
            # --- CRITICAL FIX IS HERE ---
            # Total length is l_thigh + l_calf (approx 0.426m)
            # Robot Knee Limit is approx -0.8 rads (cannot straighten fully).
            # We cap max reach at 0.38m (approx 90% extension) to stay safe.
            max_reach = 0.38  # Hard limit to prevent knee snapping
            min_reach = 0.15 
            
            l_virtual = np.clip(l_virtual, min_reach, max_reach)
            
            # --- 3. Knee Pitch (q3) ---
            cos_q3 = (l_virtual**2 - self.l_thigh**2 - self.l_calf**2) / (2 * self.l_thigh * self.l_calf)
            cos_q3 = np.clip(cos_q3, -1.0, 1.0)
            q3 = -np.arccos(cos_q3) # Maps extension to 0.0 (clamped by max_reach above)
            
            # --- 4. Hip Pitch (q2) ---
            alpha = np.arctan2(-x_eff, d_eff) 
            
            cos_beta = (l_virtual**2 + self.l_thigh**2 - self.l_calf**2) / (2 * l_virtual * self.l_thigh)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            beta = np.arccos(cos_beta)
            q2 = alpha + beta
            
            joints.extend([q1, q2, q3])
            
        return np.array(joints)
            

    def _map(self, val, min_v, max_v):
        return min_v + (val + 1) / 2 * (max_v - min_v)
        
    def reset(self):
        self.r[:] = 1.0
        self.r_dot[:] = 0.0
        self.theta = np.array([0.0, np.pi, np.pi, 0.0])