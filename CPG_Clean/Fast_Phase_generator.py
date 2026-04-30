import numpy as np

class PhaseGuidedGenerator:
    def __init__(self, robot_type='go2'):
        if robot_type == 'go2':
            self.l_hip = 0.0955
            self.l_thigh = 0.213
            self.l_calf = 0.213
            self.nominal_height = 0.28 # Lower center of mass for high speed!
            self.hip_offset_x = 0.19 
            self.hip_offset_y = 0.13
        else: # A1
            self.l_hip = 0.0838
            self.l_thigh = 0.2
            self.l_calf = 0.2
            self.nominal_height = 0.30
            self.hip_offset_x = 0.18
            self.hip_offset_y = 0.13

        self.step_height = 0.12 # Lower step height for speed (efficiency)
        self.stance_width = 0.13
        
        self.BASE_X_GAIN = 0.147 
        self.BASE_Y_GAIN = 0.110
        self.YAW_BOOST = 1.26

    def _get_phase_progress(self, phi):
        if phi >= 0:
            return (phi) / np.pi, False # Stance
        else:
            return (phi + np.pi) / np.pi, True # Swing

    def _get_velocity_gain_scaler(self, v_cmd_mag):
        v = abs(v_cmd_mag)
        
        # --- TURBO TUNING ---
        if v < 0.5:
            return np.interp(v, [0.0, 0.2, 0.5], [1.35, 1.32, 1.0])
        elif v < 1.5:
            return np.interp(v, [0.5, 0.8, 1.5], [1.0, 0.88, 0.70])
        else:
            # 1.5 m/s to 3.5 m/s
            # We drastically reduce stride gain to prevent IK explosion.
            # At 3.0 m/s, gain is 0.45. This forces the user to provide High Freq.
            return np.interp(v, [1.5, 2.5, 3.5], [0.70, 0.55, 0.45])

    def get_foot_trajectory(self, phases, command_vel, frequency=2.5, is_standing=False):
        vx, vy, wz = command_vel
        foot_positions = np.zeros((4, 3))
        leg_signs = [[1,1], [1,-1], [-1,1], [-1,-1]] 
        
        if is_standing or np.linalg.norm(command_vel) < 0.02:
            for i in range(4):
                sx, sy = leg_signs[i]
                foot_positions[i] = [0.04, sy * self.stance_width, -self.nominal_height]
            return foot_positions

        if frequency < 0.1: frequency = 0.1
        freq_scale = (2.5 / frequency) ** 0.75
        
        vx_correction = self._get_velocity_gain_scaler(vx)
        x_gain = self.BASE_X_GAIN * freq_scale * vx_correction
        y_gain = self.BASE_Y_GAIN * freq_scale 

        wz_boosted = wz * self.YAW_BOOST

        for i in range(4):
            sx, sy = leg_signs[i]
            side_sign = sy 
            
            r_x = sx * self.hip_offset_x
            r_y = sy * self.hip_offset_y
            v_leg_x = vx - (wz_boosted * r_y)
            v_leg_y = vy + (wz_boosted * r_x)
            
            # Stride Length
            step_len_x = -v_leg_x * x_gain 
            step_len_y = v_leg_y * y_gain
            
            phi = phases[i]
            p_raw, is_swing = self._get_phase_progress(phi)
            p = 6 * p_raw**5 - 15 * p_raw**4 + 10 * p_raw**3 

            if is_swing:
                xs = (2*p - 1) * step_len_x 
                ys = (2*p - 1) * step_len_y
                zs = -self.nominal_height + self.step_height * (4 * p * (1 - p))
            else:
                xs = (1 - 2*p) * step_len_x
                ys = (1 - 2*p) * step_len_y
                zs = -self.nominal_height

            ys += side_sign * self.stance_width
            foot_positions[i] = [xs + 0.04, ys, zs]
            
        return foot_positions

    def inverse_kinematics(self, foot_pos_body):
        # ... (SAME AS BEFORE) ...
        q_out = np.zeros(12)
        for i in range(4):
            x, y, z = foot_pos_body[i]
            side_sign = 1 if (i % 2 == 0) else -1
            
            yz_dist_sq = y**2 + z**2
            z_prime_sq = max(0, yz_dist_sq - self.l_hip**2)
            z_prime = -np.sqrt(z_prime_sq)
            
            q0 = np.arctan2(z, y) - np.arctan2(z_prime, side_sign * self.l_hip)
            q0 = (q0 + np.pi) % (2 * np.pi) - np.pi

            l_eff = np.sqrt(x**2 + z_prime**2)
            l_eff = np.clip(l_eff, 0.05, self.l_thigh + self.l_calf - 0.005)
            
            acos_thigh = (self.l_thigh**2 + l_eff**2 - self.l_calf**2) / (2 * self.l_thigh * l_eff)
            alpha = np.arccos(np.clip(acos_thigh, -1, 1))
            
            acos_calf = (self.l_thigh**2 + self.l_calf**2 - l_eff**2) / (2 * self.l_thigh * self.l_calf)
            gamma = np.arccos(np.clip(acos_calf, -1, 1))
            
            beta = np.arctan2(x, -z_prime)
            q1 = beta + alpha
            q2 = - (np.pi - gamma)
            
            q_out[i*3:i*3+3] = [q0, q1, q2]
        return q_out