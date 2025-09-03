"""
AI Utilities for Flappy Bird
State management, logging, and configuration utilities
"""
import csv
import os
from collections import deque


class GameStateManager:
    """Manages game state extraction and normalization"""
    
    @staticmethod
    def get_state(bird_y, bird_velocity, pipe_height, pipe_gap, pipe_x, bird_x):
        """Extract current game state"""
        gap_top = pipe_height
        gap_bottom = pipe_height + pipe_gap
        return {
            'bird_y': bird_y,
            'bird_velocity': bird_velocity,
            'gap_top': gap_top,
            'gap_bottom': gap_bottom,
            'pipe_dx': pipe_x - bird_x
        }


class ImitationLogger:
    """Logs state-action pairs for supervised learning"""
    
    def __init__(self, log_path='imitation_data.csv', ai_config=None):
        self.log_path = log_path
        self.ai_config = ai_config  # Reference to config for dynamic checking
        self.action_history = deque(maxlen=10)
        
        # Create log file if it doesn't exist (regardless of current logging state)
        if not os.path.isfile(self.log_path):
            self._create_log_file()
    
    def _create_log_file(self):
        """Create CSV file with headers"""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bird_y', 'bird_velocity', 'gap_top', 'gap_bottom', 'pipe_dx', 'action'])
    
    def log_sample(self, state, action):
        """Log a state-action pair (only if logging is enabled)"""
        # Check if logging is enabled dynamically
        if not (self.ai_config and self.ai_config.data_log):
            return
        
        self.action_history.append(action)
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                int(state['bird_y']), 
                round(state['bird_velocity'], 3),
                int(state['gap_top']), 
                int(state['gap_bottom']), 
                int(state['pipe_dx']), 
                int(action)
            ])
    
    def get_recent_actions(self):
        """Get recent action history"""
        return list(self.action_history)


class AIConfig:
    """Configuration for AI systems"""
    
    def __init__(self):
        # AI Control
        self.use_ai = True
        self.ai_mode = 'heuristic'  # 'heuristic' | 'pid' | 'plan' | 'imitation'
        
        # Data Logging
        self.data_log = True
        self.log_path = 'imitation_data.csv'
        
        # Model Loading
        self.residual_model_path = 'residual_model.pth'
        
        # Controller Parameters
        self.pid_params = {
            'kp': 0.12,
            'ki': 0.0008,
            'kd': 0.45,
            'max_integral': 300
        }
        
        self.planner_params = {
            'horizon': 40,
            'trials': 18
        }
    
    def toggle_ai(self):
        """Toggle AI on/off"""
        self.use_ai = not self.use_ai
        return self.use_ai
    
    def set_mode(self, mode):
        """Set AI mode"""
        valid_modes = ['heuristic', 'pid', 'plan', 'imitation']
        if mode in valid_modes:
            self.ai_mode = mode
            return True
        return False


def load_residual_model(model_path='residual_model.pth'):
    """Load optional residual neural network model"""
    try:
        import torch
        from ai_controllers import ResidualNet
        
        if os.path.isfile(model_path):
            model = ResidualNet()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print(f'Residual model loaded from {model_path}')
            return model
        else:
            print(f'No residual model found at {model_path}')
            return None
    except ImportError:
        print('PyTorch not available, residual model disabled')
        return None
    except Exception as e:
        print(f'Failed to load residual model: {e}')
        return None
