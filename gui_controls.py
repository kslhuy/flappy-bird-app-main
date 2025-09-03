"""
Real-time GUI controls for AI controller parameters
Seamlessly integrated with pygame Flappy Bird
"""
import pygame
import math

class Slider:
    """A slider widget for adjusting numerical parameters"""
    
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, decimal_places=3):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.decimal_places = decimal_places
        self.dragging = False
        self.handle_radius = height // 2
        
        # Colors
        self.bg_color = (60, 60, 60)
        self.track_color = (100, 100, 100)
        self.handle_color = (200, 200, 200)
        self.handle_active_color = (255, 255, 255)
        self.text_color = (255, 255, 255)
        
    def handle_event(self, event):
        """Handle mouse events for the slider"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_x, mouse_y = event.pos
                handle_x = self._get_handle_x()
                handle_rect = pygame.Rect(handle_x - self.handle_radius, 
                                        self.rect.y, 
                                        self.handle_radius * 2, 
                                        self.rect.height)
                if handle_rect.collidepoint(mouse_x, mouse_y):
                    self.dragging = True
                elif self.rect.collidepoint(mouse_x, mouse_y):
                    # Click on track - jump to position
                    self._update_value_from_mouse(mouse_x)
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._update_value_from_mouse(event.pos[0])
    
    def _get_handle_x(self):
        """Get the x position of the handle based on current value"""
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + ratio * self.rect.width
    
    def _update_value_from_mouse(self, mouse_x):
        """Update value based on mouse position"""
        ratio = (mouse_x - self.rect.x) / self.rect.width
        ratio = max(0, min(1, ratio))  # Clamp to [0, 1]
        self.val = self.min_val + ratio * (self.max_val - self.min_val)
    
    def draw(self, screen, font):
        """Draw the slider"""
        # Background
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=5)
        
        # Track
        track_rect = pygame.Rect(self.rect.x + 5, 
                               self.rect.centery - 2, 
                               self.rect.width - 10, 4)
        pygame.draw.rect(screen, self.track_color, track_rect, border_radius=2)
        
        # Handle
        handle_x = self._get_handle_x()
        handle_color = self.handle_active_color if self.dragging else self.handle_color
        pygame.draw.circle(screen, handle_color, 
                         (int(handle_x), self.rect.centery), 
                         self.handle_radius)
        
        # Label and value text
        value_str = f"{self.val:.{self.decimal_places}f}"
        text = f"{self.label}: {value_str}"
        text_surface = font.render(text, True, self.text_color)
        screen.blit(text_surface, (self.rect.x, self.rect.y - 20))


class Button:
    """A simple button widget"""
    
    def __init__(self, x, y, width, height, text, callback=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.pressed = False
        
        # Colors
        self.bg_color = (70, 130, 180)
        self.bg_hover_color = (100, 149, 237)
        self.bg_pressed_color = (30, 80, 140)
        self.text_color = (255, 255, 255)
    
    def handle_event(self, event):
        """Handle mouse events for the button"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.pressed = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if self.pressed and self.rect.collidepoint(event.pos):
                    if self.callback:
                        self.callback()
                self.pressed = False
    
    def draw(self, screen, font):
        """Draw the button"""
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos)
        
        if self.pressed:
            color = self.bg_pressed_color
        elif is_hover:
            color = self.bg_hover_color
        else:
            color = self.bg_color
            
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        # Text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


class ControlPanel:
    """Main control panel for AI parameter tuning"""
    
    def __init__(self, x, y, width, height, controllers, ai_config):
        self.rect = pygame.Rect(x, y, width, height)
        self.controllers = controllers
        self.ai_config = ai_config
        self.visible = True  # Always visible now
        self.font = pygame.font.SysFont('Arial', 12)
        self.title_font = pygame.font.SysFont('Arial', 14, bold=True)
        
        # Colors
        self.bg_color = (30, 30, 30)  # Solid background
        self.border_color = (100, 100, 100)
        self.text_color = (255, 255, 255)
        
        self.sliders = {}
        self.buttons = {}
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all sliders and buttons"""
        start_y = self.rect.y + 30
        slider_height = 20
        spacing = 35
        slider_width = self.rect.width - 20
        
        # Heuristic Controller sliders
        current_y = start_y
        self.sliders['heuristic_margin_base'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0.0, 20.0, 6.0, "Base Margin", 1
        )
        current_y += spacing
        
        self.sliders['heuristic_velocity_factor'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0.0, 2.0, 0.9, "Velocity Factor", 2
        )
        current_y += spacing
        
        self.sliders['heuristic_approach_distance'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            50, 300, 180, "Approach Distance", 0
        )
        current_y += spacing + 10
        
        # PID Controller sliders
        self.sliders['pid_kp'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0.0, 0.5, 0.12, "PID Kp", 3
        )
        current_y += spacing
        
        self.sliders['pid_ki'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0.0, 0.01, 0.0008, "PID Ki", 4
        )
        current_y += spacing
        
        self.sliders['pid_kd'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0.0, 1.0, 0.45, "PID Kd", 3
        )
        current_y += spacing
        
        self.sliders['pid_control_threshold'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0, 20, 5, "Control Threshold", 0
        )
        current_y += spacing
        
        self.sliders['pid_velocity_threshold'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0, 15, 5, "Velocity Threshold", 0
        )
        current_y += spacing
        
        self.sliders['pid_approach_distance'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            50, 300, 200, "Approach Distance", 0
        )
        current_y += spacing + 10
        
        # Planner Controller sliders
        self.sliders['planner_horizon'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            10, 100, 40, "Horizon", 0
        )
        current_y += spacing
        
        self.sliders['planner_trials'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            5, 50, 18, "Trials", 0
        )
        current_y += spacing
        
        self.sliders['planner_flap_probability'] = Slider(
            self.rect.x + 10, current_y, slider_width, slider_height,
            0.0, 0.5, 0.15, "Flap Probability", 3
        )
        
        # Reset buttons
        reset_y = current_y + spacing + 10
        button_width = (slider_width - 20) // 3
        self.buttons['reset_heuristic'] = Button(
            self.rect.x + 10, reset_y, button_width, 25, "Reset H",
            lambda: self._reset_controller_params('heuristic')
        )
        self.buttons['reset_pid'] = Button(
            self.rect.x + 20 + button_width, reset_y, button_width, 25, "Reset P",
            lambda: self._reset_controller_params('pid')
        )
        self.buttons['reset_planner'] = Button(
            self.rect.x + 30 + button_width * 2, reset_y, button_width, 25, "Reset Pl",
            lambda: self._reset_controller_params('planner')
        )
    
    def _toggle_visibility(self):
        """Toggle panel visibility"""
        self.visible = not self.visible
        self.buttons['toggle'].text = "Hide" if self.visible else "Show"
    
    def _reset_controller_params(self, controller_type):
        """Reset parameters for a specific controller type"""
        if controller_type == 'heuristic':
            self.sliders['heuristic_margin_base'].val = 6.0
            self.sliders['heuristic_velocity_factor'].val = 0.9
            self.sliders['heuristic_approach_distance'].val = 180
        elif controller_type == 'pid':
            self.sliders['pid_kp'].val = 0.12
            self.sliders['pid_ki'].val = 0.0008
            self.sliders['pid_kd'].val = 0.45
            self.sliders['pid_control_threshold'].val = 5
            self.sliders['pid_velocity_threshold'].val = 5
            self.sliders['pid_approach_distance'].val = 200
        elif controller_type == 'planner':
            self.sliders['planner_horizon'].val = 40
            self.sliders['planner_trials'].val = 18
            self.sliders['planner_flap_probability'].val = 0.15
    
    def handle_event(self, event):
        """Handle events for all widgets"""
        # Handle all sliders and buttons
        for slider in self.sliders.values():
            slider.handle_event(event)
        
        for button in self.buttons.values():
            button.handle_event(event)
        
        # Update controller parameters in real-time
        self._update_controller_params()
    
    def _update_controller_params(self):
        """Update controller parameters based on slider values"""
        # Update PID controller
        if 'pid' in self.controllers:
            pid = self.controllers['pid']
            pid.kp = self.sliders['pid_kp'].val
            pid.ki = self.sliders['pid_ki'].val
            pid.kd = self.sliders['pid_kd'].val
        
        # Update Planner controller
        if 'plan' in self.controllers:
            planner = self.controllers['plan']
            planner.horizon = int(self.sliders['planner_horizon'].val)
            planner.trials = int(self.sliders['planner_trials'].val)
    
    def draw(self, screen):
        """Draw the control panel"""
        # Draw solid background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 2)
        
        # Title
        title = f"AI Controls - {self.ai_config.ai_mode.upper()}"
        title_surface = self.title_font.render(title, True, self.text_color)
        screen.blit(title_surface, (self.rect.x + 10, self.rect.y + 5))
        
        # Section headers
        y_offset = 35
        sections = [
            ("HEURISTIC", 0),
            ("PID", 3 * 35 + 10),
            ("PLANNER", 9 * 35 + 20)
        ]
        
        for section_name, offset in sections:
            section_surface = self.font.render(section_name, True, (255, 200, 100))
            screen.blit(section_surface, (self.rect.x + 10, self.rect.y + y_offset + offset - 15))
        
        # Draw all sliders
        relevant_sliders = self._get_relevant_sliders()
        for name, slider in self.sliders.items():
            # Highlight relevant sliders for current AI mode
            if name in relevant_sliders:
                slider.handle_color = (100, 255, 100)  # Green for active
                slider.handle_active_color = (150, 255, 150)
            else:
                slider.handle_color = (150, 150, 150)  # Gray for inactive
                slider.handle_active_color = (200, 200, 200)
            slider.draw(screen, self.font)
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(screen, self.font)
    
    def _get_relevant_sliders(self):
        """Get slider names relevant to current AI mode"""
        mode = self.ai_config.ai_mode
        if mode == 'heuristic':
            return ['heuristic_margin_base', 'heuristic_velocity_factor', 'heuristic_approach_distance']
        elif mode == 'pid':
            return ['pid_kp', 'pid_ki', 'pid_kd', 'pid_control_threshold', 'pid_velocity_threshold', 'pid_approach_distance']
        elif mode == 'plan':
            return ['planner_horizon', 'planner_trials', 'planner_flap_probability']
        return []
    
    def get_heuristic_params(self):
        """Get current heuristic parameters"""
        return {
            'margin_base': self.sliders['heuristic_margin_base'].val,
            'velocity_factor': self.sliders['heuristic_velocity_factor'].val,
            'approach_distance': self.sliders['heuristic_approach_distance'].val
        }
    
    def get_pid_params(self):
        """Get current PID parameters"""
        return {
            'control_threshold': self.sliders['pid_control_threshold'].val,
            'velocity_threshold': self.sliders['pid_velocity_threshold'].val,
            'approach_distance': self.sliders['pid_approach_distance'].val
        }
    
    def get_planner_params(self):
        """Get current planner parameters"""
        return {
            'flap_probability': self.sliders['planner_flap_probability'].val
        }
