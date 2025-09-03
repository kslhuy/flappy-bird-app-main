"""
Heuristic Controller Tuner GUI
Interactive parameter tuning for Flappy Bird AI with real-time visualization
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pygame
import random
import threading
import time
import json
import os
from ai_controllers import HeuristicController, ResidualNet
from ai_utils import GameStateManager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class HeuristicTunerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Heuristic Controller Tuner - Flappy Bird AI")
        self.root.geometry("1200x800")
        
        # Controller parameters
        self.params = {
            'base_margin': 6.0,          # Base flap threshold
            'velocity_factor': 0.9,      # How much falling velocity affects margin
            'approach_distance': 180,    # Distance when pipe is "approaching"
            'velocity_threshold': 2.0,   # Velocity considered "falling fast"
            'gap_bias': 0.0,            # Bias toward top/bottom of gap (-1 to 1)
            'early_flap': False,        # Flap earlier when approaching
            'conservative_mode': False,  # More conservative flapping
        }
        
        # Performance tracking
        self.performance_history = []
        self.current_score = 0
        self.games_played = 0
        
        # Game simulation thread
        self.simulation_running = False
        self.sim_thread = None
        
        self.setup_gui()
        self.load_presets()
        
    def setup_gui(self):
        # Main frame layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Heuristic Controller Parameters", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel - Visualization and explanation
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_controls(left_frame)
        self.setup_visualization(right_frame)
        self.setup_explanation()
        
    def setup_controls(self, parent):
        """Setup parameter control widgets"""
        self.param_vars = {}
        self.param_widgets = {}
        
        # Parameter definitions with explanations
        param_config = {
            'base_margin': {
                'label': 'Base Margin',
                'range': (0.0, 20.0),
                'step': 0.5,
                'tooltip': 'Base distance below gap center before flapping. Lower = more aggressive.'
            },
            'velocity_factor': {
                'label': 'Velocity Factor', 
                'range': (0.0, 2.0),
                'step': 0.1,
                'tooltip': 'How much falling speed increases margin. Higher = more reactive to falling.'
            },
            'approach_distance': {
                'label': 'Approach Distance',
                'range': (50, 300),
                'step': 10,
                'tooltip': 'Distance when pipe is considered "approaching". Lower = flap later.'
            },
            'velocity_threshold': {
                'label': 'Velocity Threshold',
                'range': (0.0, 10.0), 
                'step': 0.5,
                'tooltip': 'Velocity considered "falling fast". Lower = more sensitive to speed.'
            },
            'gap_bias': {
                'label': 'Gap Bias',
                'range': (-1.0, 1.0),
                'step': 0.1, 
                'tooltip': 'Aim offset in gap: -1=top, 0=center, 1=bottom.'
            }
        }
        
        row = 0
        for param, config in param_config.items():
            # Parameter label
            label = ttk.Label(parent, text=config['label'])
            label.grid(row=row, column=0, sticky=tk.W, pady=2)
            
            # Value variable
            var = tk.DoubleVar(value=self.params[param])
            self.param_vars[param] = var
            
            # Scale widget
            scale = ttk.Scale(
                parent, 
                from_=config['range'][0], 
                to=config['range'][1],
                variable=var,
                command=lambda val, p=param: self.on_param_change(p, val)
            )
            scale.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            
            # Value display
            value_label = ttk.Label(parent, text=f"{self.params[param]:.1f}")
            value_label.grid(row=row, column=2, sticky=tk.W, pady=2)
            self.param_widgets[param] = value_label
            
            # Tooltip
            self.create_tooltip(scale, config['tooltip'])
            
            row += 1
        
        # Boolean parameters
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, columnspan=3, sticky=tk.EW, pady=10)
        row += 1
        
        # Early flap checkbox
        self.early_flap_var = tk.BooleanVar(value=self.params['early_flap'])
        early_check = ttk.Checkbutton(
            parent, 
            text="Early Flap Mode", 
            variable=self.early_flap_var,
            command=lambda: self.on_bool_change('early_flap', self.early_flap_var.get())
        )
        early_check.grid(row=row, columnspan=2, sticky=tk.W, pady=2)
        self.create_tooltip(early_check, "Flap earlier when pipe approaches")
        row += 1
        
        # Conservative mode
        self.conservative_var = tk.BooleanVar(value=self.params['conservative_mode'])
        cons_check = ttk.Checkbutton(
            parent,
            text="Conservative Mode",
            variable=self.conservative_var, 
            command=lambda: self.on_bool_change('conservative_mode', self.conservative_var.get())
        )
        cons_check.grid(row=row, columnspan=2, sticky=tk.W, pady=2)
        self.create_tooltip(cons_check, "More cautious flapping behavior")
        row += 1
        
        # Control buttons
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, columnspan=3, sticky=tk.EW, pady=10)
        row += 1
        
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, columnspan=3, sticky=tk.EW, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="Start Simulation", command=self.toggle_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="Reset Parameters", command=self.reset_parameters).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Save Preset", command=self.save_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Load Preset", command=self.load_preset).pack(side=tk.LEFT, padx=2)
        
        # Performance display
        row += 1
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, columnspan=3, sticky=tk.EW, pady=10)
        row += 1
        
        perf_frame = ttk.LabelFrame(parent, text="Performance", padding=5)
        perf_frame.grid(row=row, columnspan=3, sticky=tk.EW, pady=5)
        
        self.score_label = ttk.Label(perf_frame, text="Current Score: 0")
        self.score_label.pack(anchor=tk.W)
        
        self.games_label = ttk.Label(perf_frame, text="Games Played: 0")
        self.games_label.pack(anchor=tk.W)
        
        self.avg_label = ttk.Label(perf_frame, text="Average Score: 0.0")
        self.avg_label.pack(anchor=tk.W)
        
        # Configure grid weights
        parent.columnconfigure(1, weight=1)
        
    def setup_visualization(self, parent):
        """Setup performance visualization"""
        viz_frame = ttk.LabelFrame(parent, text="Performance Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.tight_layout(pad=3.0)
        
        # Score history plot
        self.ax1.set_title("Score History")
        self.ax1.set_xlabel("Game")
        self.ax1.set_ylabel("Score")
        self.ax1.grid(True, alpha=0.3)
        
        # Parameter effect plot
        self.ax2.set_title("Parameter Impact Analysis")
        self.ax2.set_xlabel("Parameter Value")
        self.ax2.set_ylabel("Average Score")
        self.ax2.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_explanation(self):
        """Setup explanation text"""
        explanation = """
HEURISTIC CONTROLLER CONCEPTS:

The heuristic controller uses rule-based logic to control the bird:

1. GAP TARGETING: Aims for the center of the pipe gap (with optional bias)

2. DYNAMIC MARGIN: Adjusts flap threshold based on:
   - Base margin: Fixed distance below target before flapping
   - Velocity factor: Increases margin when falling fast
   - Approach distance: Only acts when pipe is close enough

3. KEY PARAMETERS TO TUNE:

   • Base Margin (0-20): Core flap threshold
     - Lower = more aggressive, higher scores but crash risk
     - Higher = safer but may miss gaps
   
   • Velocity Factor (0-2): Falling speed response  
     - Higher = more reactive to speed changes
     - Lower = ignores velocity, consistent behavior
   
   • Approach Distance (50-300): When to start caring about pipe
     - Lower = flap later, more precise timing
     - Higher = flap earlier, safer but inefficient
   
   • Gap Bias (-1 to 1): Where in gap to aim
     - -1 = aim for top of gap
     - 0 = aim for center  
     - 1 = aim for bottom

4. TUNING STRATEGY:
   - Start with default values
   - Adjust base_margin first (most important)
   - Fine-tune velocity_factor for speed response
   - Optimize approach_distance for timing
   - Use gap_bias to avoid obstacles

5. PERFORMANCE INDICATORS:
   - Score: How many pipes passed
   - Consistency: Low variance in scores
   - Efficiency: Minimal unnecessary flaps
        """
        
        # Add explanation text widget in a separate window or frame
        self.explanation_window = None
        
    def show_explanation(self):
        if self.explanation_window is None or not self.explanation_window.winfo_exists():
            self.explanation_window = tk.Toplevel(self.root)
            self.explanation_window.title("Heuristic Controller Guide")
            self.explanation_window.geometry("600x500")
            
            text_widget = tk.Text(self.explanation_window, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            explanation = """HEURISTIC CONTROLLER CONCEPTS:

The heuristic controller uses rule-based logic to control the bird:

1. GAP TARGETING: Aims for the center of the pipe gap (with optional bias)

2. DYNAMIC MARGIN: Adjusts flap threshold based on:
   - Base margin: Fixed distance below target before flapping
   - Velocity factor: Increases margin when falling fast
   - Approach distance: Only acts when pipe is close enough

3. KEY PARAMETERS TO TUNE:

   • Base Margin (0-20): Core flap threshold
     - Lower = more aggressive, higher scores but crash risk
     - Higher = safer but may miss gaps
   
   • Velocity Factor (0-2): Falling speed response  
     - Higher = more reactive to speed changes
     - Lower = ignores velocity, consistent behavior
   
   • Approach Distance (50-300): When to start caring about pipe
     - Lower = flap later, more precise timing
     - Higher = flap earlier, safer but inefficient
   
   • Gap Bias (-1 to 1): Where in gap to aim
     - -1 = aim for top of gap
     - 0 = aim for center  
     - 1 = aim for bottom

4. TUNING STRATEGY:
   - Start with default values
   - Adjust base_margin first (most important)
   - Fine-tune velocity_factor for speed response
   - Optimize approach_distance for timing
   - Use gap_bias to avoid obstacles

5. PERFORMANCE INDICATORS:
   - Score: How many pipes passed
   - Consistency: Low variance in scores
   - Efficiency: Minimal unnecessary flaps"""
            
            text_widget.insert(tk.END, explanation)
            text_widget.config(state=tk.DISABLED)
        else:
            self.explanation_window.lift()
    
    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="yellow", font=("Arial", 9))
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
    
    def on_param_change(self, param, value):
        """Handle parameter change"""
        self.params[param] = float(value)
        self.param_widgets[param].config(text=f"{float(value):.1f}")
        
    def on_bool_change(self, param, value):
        """Handle boolean parameter change"""
        self.params[param] = value
        
    def toggle_simulation(self):
        """Start/stop simulation"""
        if not self.simulation_running:
            self.simulation_running = True
            self.start_btn.config(text="Stop Simulation")
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()
        else:
            self.simulation_running = False
            self.start_btn.config(text="Start Simulation")
            
    def run_simulation(self):
        """Run game simulation with current parameters"""
        try:
            # Initialize minimal pygame for simulation
            pygame.init()
            
            while self.simulation_running:
                score = self.simulate_single_game()
                self.performance_history.append(score)
                self.games_played += 1
                self.current_score = score
                
                # Update GUI in main thread
                self.root.after(0, self.update_performance_display)
                self.root.after(0, self.update_plots)
                
                time.sleep(0.1)  # Brief pause between games
                
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Error in simulation: {e}")
            self.simulation_running = False
            self.root.after(0, lambda: self.start_btn.config(text="Start Simulation"))
    
    def simulate_single_game(self):
        """Simulate a single game and return score"""
        # Simplified game simulation
        # This would integrate with your actual game logic
        
        # Create enhanced heuristic controller
        controller = EnhancedHeuristicController(self.params)
        
        # Simulate game state
        bird_y = 300
        bird_velocity = 0
        score = 0
        
        for frame in range(1000):  # Max 1000 frames per game
            # Simulate pipe
            pipe_x = 400 - (frame * 3) % 500
            pipe_height = random.randint(100, 350)
            gap_top = pipe_height
            gap_bottom = pipe_height + 150
            
            # Create state
            state = {
                'bird_y': bird_y,
                'bird_velocity': bird_velocity,
                'gap_top': gap_top,
                'gap_bottom': gap_bottom,
                'pipe_dx': pipe_x - 100  # bird at x=100
            }
            
            # Get controller decision
            action = controller.decide(state, bird_height=24)
            
            # Apply physics
            if action:
                bird_velocity = -10  # flap power
            bird_velocity += 0.5  # gravity
            bird_y += bird_velocity
            
            # Check collision
            if bird_y < 0 or bird_y > 576:  # hit ceiling/floor
                break
                
            # Check pipe collision (simplified)
            if 80 < pipe_x < 120:  # bird in pipe area
                bird_center = bird_y + 12
                if not (gap_top < bird_center < gap_bottom):
                    break
                    
            # Score point
            if pipe_x == 97:  # just passed pipe
                score += 1
                
        return score
    
    def update_performance_display(self):
        """Update performance labels"""
        self.score_label.config(text=f"Current Score: {self.current_score}")
        self.games_label.config(text=f"Games Played: {self.games_played}")
        
        if self.performance_history:
            avg_score = sum(self.performance_history) / len(self.performance_history)
            self.avg_label.config(text=f"Average Score: {avg_score:.1f}")
    
    def update_plots(self):
        """Update visualization plots"""
        if not self.performance_history:
            return
            
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Score history
        self.ax1.plot(self.performance_history, 'b-', alpha=0.7)
        if len(self.performance_history) > 10:
            # Moving average
            window = min(10, len(self.performance_history))
            ma = np.convolve(self.performance_history, np.ones(window)/window, mode='valid')
            self.ax1.plot(range(window-1, len(self.performance_history)), ma, 'r-', linewidth=2, label='Moving Average')
            self.ax1.legend()
        
        self.ax1.set_title("Score History")
        self.ax1.set_xlabel("Game")
        self.ax1.set_ylabel("Score")
        self.ax1.grid(True, alpha=0.3)
        
        # Parameter impact (show recent performance vs parameter value)
        if len(self.performance_history) > 5:
            recent_avg = sum(self.performance_history[-5:]) / 5
            self.ax2.bar(['Current'], [recent_avg], color='green', alpha=0.7)
            self.ax2.set_title(f"Recent Performance (Base Margin: {self.params['base_margin']:.1f})")
            self.ax2.set_ylabel("Average Score (last 5 games)")
        
        self.canvas.draw()
    
    def reset_parameters(self):
        """Reset to default parameters"""
        defaults = {
            'base_margin': 6.0,
            'velocity_factor': 0.9,
            'approach_distance': 180,
            'velocity_threshold': 2.0,
            'gap_bias': 0.0,
            'early_flap': False,
            'conservative_mode': False,
        }
        
        for param, value in defaults.items():
            if param in self.param_vars:
                self.param_vars[param].set(value)
                self.params[param] = value
                self.param_widgets[param].config(text=f"{value:.1f}")
            elif param == 'early_flap':
                self.early_flap_var.set(value)
                self.params[param] = value
            elif param == 'conservative_mode':
                self.conservative_var.set(value)
                self.params[param] = value
    
    def save_preset(self):
        """Save current parameters as preset"""
        name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        if name:
            presets = self.load_presets()
            presets[name] = self.params.copy()
            with open('heuristic_presets.json', 'w') as f:
                json.dump(presets, f, indent=2)
            messagebox.showinfo("Success", f"Preset '{name}' saved!")
    
    def load_preset(self):
        """Load preset parameters"""
        presets = self.load_presets()
        if not presets:
            messagebox.showinfo("No Presets", "No presets found.")
            return
            
        # Create selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Load Preset")
        selection_window.geometry("300x200")
        
        tk.Label(selection_window, text="Select preset:").pack(pady=10)
        
        preset_var = tk.StringVar()
        preset_list = ttk.Combobox(selection_window, textvariable=preset_var, values=list(presets.keys()))
        preset_list.pack(pady=10)
        
        def load_selected():
            selected = preset_var.get()
            if selected in presets:
                self.params.update(presets[selected])
                # Update GUI
                for param, value in self.params.items():
                    if param in self.param_vars:
                        self.param_vars[param].set(value)
                        self.param_widgets[param].config(text=f"{value:.1f}")
                    elif param == 'early_flap':
                        self.early_flap_var.set(value)
                    elif param == 'conservative_mode':
                        self.conservative_var.set(value)
                selection_window.destroy()
                messagebox.showinfo("Success", f"Preset '{selected}' loaded!")
        
        ttk.Button(selection_window, text="Load", command=load_selected).pack(pady=10)
    
    def load_presets(self):
        """Load presets from file"""
        try:
            if os.path.exists('heuristic_presets.json'):
                with open('heuristic_presets.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading presets: {e}")
        return {}
    
    def run(self):
        """Start the GUI"""
        # Add menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Controller Guide", command=self.show_explanation)
        
        self.root.mainloop()


class EnhancedHeuristicController:
    """Enhanced heuristic controller with tunable parameters"""
    
    def __init__(self, params):
        self.params = params
    
    def decide(self, state, bird_height):
        """Make flap decision based on current parameters"""
        gap_center = (state['gap_top'] + state['gap_bottom']) / 2
        
        # Apply gap bias
        target_y = gap_center + self.params['gap_bias'] * 75  # 75 = half gap size
        
        bird_center = state['bird_y'] + bird_height / 2
        vertical_error = bird_center - target_y
        
        # Dynamic margin calculation
        margin = self.params['base_margin']
        
        # Velocity adjustment
        if state['bird_velocity'] > self.params['velocity_threshold']:
            margin += state['bird_velocity'] * self.params['velocity_factor']
        
        # Conservative mode adjustment
        if self.params['conservative_mode']:
            margin *= 1.3
        
        # Early flap mode
        if self.params['early_flap']:
            margin *= 0.8
        
        # Approach distance check
        approaching = state['pipe_dx'] < self.params['approach_distance']
        
        # Decision logic
        should_flap = vertical_error > margin and approaching
        
        return 1 if should_flap else 0


if __name__ == "__main__":
    try:
        import tkinter.simpledialog
        app = HeuristicTunerGUI()
        app.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install matplotlib")
