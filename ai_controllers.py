"""
AI Controllers for Flappy Bird
Contains heuristic, PID, planner-based, and imitation learning control algorithms
"""
import random
import math
import os
try:
    import torch
    import torch.nn as nn
    import pickle
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
        # Default parameters (can be overridden by GUI)
        self.margin_base = 6.0
        self.velocity_factor = 0.9
        self.approach_distance = 180
    
    def set_params(self, margin_base=None, velocity_factor=None, approach_distance=None):
        """Update parameters from GUI"""
        if margin_base is not None:
            self.margin_base = margin_base
        if velocity_factor is not None:
            self.velocity_factor = velocity_factor
        if approach_distance is not None:
            self.approach_distance = approach_distance
    
    def decide(self, state, bird_height, screen_width, screen_height):
        gap_center = (state['gap_top'] + state['gap_bottom']) / 2
        bird_center = state['bird_y'] + bird_height / 2
        vertical_error = bird_center - gap_center
        
        # Dynamic threshold: more aggressive if falling fast
        margin = self.margin_base + max(0, state['bird_velocity']) * self.velocity_factor
        
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
        approaching = state['pipe_dx'] < self.approach_distance
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
        # Additional parameters for GUI control
        self.control_threshold = 5
        self.velocity_threshold = 5
        self.approach_distance = 200
    
    def set_params(self, control_threshold=None, velocity_threshold=None, approach_distance=None):
        """Update parameters from GUI"""
        if control_threshold is not None:
            self.control_threshold = control_threshold
        if velocity_threshold is not None:
            self.velocity_threshold = velocity_threshold
        if approach_distance is not None:
            self.approach_distance = approach_distance
    
    def decide(self, state, bird_height):
        gap_center = (state['gap_top'] + state['gap_bottom']) / 2
        bird_center = state['bird_y'] + bird_height / 2
        error = gap_center - bird_center  # positive = need to go down, negative = need to rise
        
        self.integral += error
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = error - self.last_error
        self.last_error = error
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Map control to flap decision (using GUI parameters)
        need_lift = control < -self.control_threshold or state['bird_velocity'] > self.velocity_threshold
        approaching = state['pipe_dx'] < self.approach_distance
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
        # Additional parameter for GUI control
        self.flap_probability = 0.15
    
    def set_params(self, flap_probability=None):
        """Update parameters from GUI"""
        if flap_probability is not None:
            self.flap_probability = flap_probability
    
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
                # Random future policy with occasional flaps (using GUI parameter)
                for t in range(1, self.horizon):
                    if t % 10 == 0:
                        seq.append(1 if random.random() < self.flap_probability else 0)
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


class ImitationController:
    """Neural network controller trained on expert demonstrations"""
    
    def __init__(self, model_path='flappy_bird_imitation_model.pth'):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        
        if torch and os.path.exists(model_path):
            try:
                self._load_model(model_path)
                print(f"Imitation model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load imitation model: {e}")
        else:
            print(f"Imitation model not found at {model_path}")
    
    def _load_model(self, model_path):
        """Load trained neural network model"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Recreate model architecture
        self.model = ImitationNet()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler for input normalization
        self.scaler = checkpoint['scaler']
        self.model_loaded = True
    
    def decide(self, state, bird_height):
        """Make decision using trained neural network"""
        if not self.model_loaded:
            # Fallback to simple heuristic if model not loaded
            gap_center = (state['gap_top'] + state['gap_bottom']) / 2
            bird_center = state['bird_y'] + bird_height / 2
            return 1 if bird_center > gap_center + 20 else 0
        
        # Prepare input features
        features = [
            state['bird_y'],
            state['bird_velocity'],
            state['gap_top'],
            state['gap_bottom'],
            state['pipe_dx']
        ]
        
        # Normalize features using saved scaler
        features_normalized = self.scaler.transform([features])[0]
        
        # Make prediction
        with torch.no_grad():
            inputs = torch.FloatTensor(features_normalized)
            flap_probability = self.model(inputs).item()
        
        # Decision threshold
        return 1 if flap_probability > 0.5 else 0


class ImitationNet(nn.Module if torch else object):
    """Neural network architecture for imitation learning"""
    
    def __init__(self, input_size=5, hidden_sizes=[64, 32, 16]):
        if not torch:
            return
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


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
    elif mode == 'imitation':
        return ImitationController(kwargs.get('model_path', 'flappy_bird_imitation_model.pth'))
    else:
        raise ValueError(f"Unknown AI mode: {mode}")
