"""
AI Controllers for Flappy Bird
Contains heuristic, PID, and planner-based control algorithms
"""
import random
import math
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = object


class ResidualNet(nn.Module if torch else object):
    """Optional neural network to adjust heuristic threshold"""
    def __init__(self):
        if not torch:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


class HeuristicController:
    """Rule-based controller targeting gap center with dynamic margin"""
    
    def __init__(self, residual_model=None):
        self.residual_model = residual_model
    
    def decide(self, state, bird_height, screen_width, screen_height):
        gap_center = (state['gap_top'] + state['gap_bottom']) / 2
        bird_center = state['bird_y'] + bird_height / 2
        vertical_error = bird_center - gap_center
        
        # Dynamic threshold: more aggressive if falling fast
        margin = 6 + max(0, state['bird_velocity']) * 0.9
        
        # Residual adjustment from neural network
        if self.residual_model and torch:
            with torch.no_grad():
                inp = torch.tensor([
                    bird_center / screen_height,
                    state['bird_velocity'] / 15.0,
                    gap_center / screen_height,
                    state['gap_top'] / screen_height,
                    state['pipe_dx'] / screen_width
                ], dtype=torch.float32)
                residual = float(self.residual_model(inp).item()) * 10
                margin += residual
        
        # Flap when too far below gap center and pipe is approaching
        approaching = state['pipe_dx'] < 180
        return 1 if (vertical_error > margin and approaching) else 0


class PIDController:
    """PID controller maintaining bird position relative to gap center"""
    
    def __init__(self, kp=0.12, ki=0.0008, kd=0.45, max_integral=300):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.integral = 0.0
        self.last_error = 0.0
    
    def decide(self, state, bird_height):
        gap_center = (state['gap_top'] + state['gap_bottom']) / 2
        bird_center = state['bird_y'] + bird_height / 2
        error = gap_center - bird_center  # positive = need to go down, negative = need to rise
        
        self.integral += error
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = error - self.last_error
        self.last_error = error
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Map control to flap decision
        need_lift = control < -5 or state['bird_velocity'] > 5
        approaching = state['pipe_dx'] < 200
        return 1 if (need_lift and approaching) else 0
    
    def reset(self):
        """Reset PID state for new game"""
        self.integral = 0.0
        self.last_error = 0.0


class PlannerController:
    """Model-based controller using physics simulation"""
    
    def __init__(self, horizon=40, trials=18, gravity=0.5, flap_power=-10):
        self.horizon = horizon
        self.trials = trials
        self.gravity = gravity
        self.flap_power = flap_power
    
    def simulate_vertical(self, y, vel, action_sequence):
        """Simple physics simulation"""
        v = vel
        pos = y
        for a in action_sequence:
            if a:  # flap
                v = self.flap_power
            v += self.gravity
            pos += v
        return pos, v
    
    def decide(self, state, bird_height):
        gap_center = (state['gap_top'] + state['gap_bottom']) / 2
        best_choice = 0
        best_score = float('inf')
        
        # Try both first actions (flap vs no-flap)
        for first_action in [0, 1]:
            accum = 0
            for _ in range(self.trials):
                seq = [first_action]
                # Random future policy with occasional flaps
                for t in range(1, self.horizon):
                    if t % 10 == 0:
                        seq.append(1 if random.random() < 0.15 else 0)
                    else:
                        seq.append(0)
                
                final_y, _ = self.simulate_vertical(state['bird_y'], state['bird_velocity'], seq)
                final_center = final_y + bird_height / 2
                accum += abs(final_center - gap_center)
            
            avg_err = accum / self.trials
            if avg_err < best_score:
                best_score = avg_err
                best_choice = first_action
        
        return best_choice


# Controller factory
def create_controller(mode, **kwargs):
    """Factory function to create AI controllers"""
    if mode == 'heuristic':
        return HeuristicController(kwargs.get('residual_model'))
    elif mode == 'pid':
        return PIDController(
            kp=kwargs.get('kp', 0.12),
            ki=kwargs.get('ki', 0.0008), 
            kd=kwargs.get('kd', 0.45),
            max_integral=kwargs.get('max_integral', 300)
        )
    elif mode == 'plan':
        return PlannerController(
            horizon=kwargs.get('horizon', 40),
            trials=kwargs.get('trials', 18),
            gravity=kwargs.get('gravity', 0.5),
            flap_power=kwargs.get('flap_power', -10)
        )
    else:
        raise ValueError(f"Unknown AI mode: {mode}")
