import numpy as np

class EnhancedHopfOscillator:
    def __init__(self, dt, alpha=50.0, mu=1.0, omega=2*np.pi*1.5):
        self.dt = dt
        self.alpha = alpha # Convergence rate
        self.mu = mu       # Amplitude^2
        self.omega = omega # Frequency (rad/s)
        
        # State [x, y]
        self.x, self.y = 1.0, 0.0
        self.phase_offset = 0.0

    def reset(self):
        self.x, self.y = 1.0, 0.0

    def step(self):
        # Limit Cycle Dynamics (Hopf)
        # dx = alpha*(mu - r^2)*x - omega*y
        # dy = alpha*(mu - r^2)*y + omega*x
        
        r2 = self.x**2 + self.y**2
        dx = self.alpha * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.alpha * (self.mu - r2) * self.y + self.omega * self.x
        
        # Coupling/Offset Logic (Simplified)
        # We rotate the output by the phase_offset for the consumer
        # Actual state integration:
        self.x += dx * self.dt
        self.y += dy * self.dt
        
        # Apply Phase Offset rotation for output
        # x_out = x*cos(off) - y*sin(off)
        # y_out = x*sin(off) + y*cos(off)
        c, s = np.cos(self.phase_offset), np.sin(self.phase_offset)
        x_out = self.x * c - self.y * s
        y_out = self.x * s + self.y * c
        
        return x_out, y_out