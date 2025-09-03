"""
Simple launcher for the Heuristic Tuner GUI
Handles missing dependencies gracefully
"""
import sys

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    return missing

def install_dependencies(packages):
    """Provide installation instructions"""
    print("Missing required packages:")
    for pkg in packages:
        print(f"  - {pkg}")
    
    print("\nTo install missing packages, run:")
    if "matplotlib" in packages or "numpy" in packages:
        print("  pip install matplotlib numpy")
    
    print("\nNote: tkinter usually comes with Python. If missing, you may need to reinstall Python.")

def main():
    missing = check_dependencies()
    
    if missing:
        install_dependencies(missing)
        return
    
    try:
        from heuristic_tuner import HeuristicTunerGUI
        app = HeuristicTunerGUI()
        app.run()
    except Exception as e:
        print(f"Error starting tuner: {e}")
        print("\nTrying simplified version without plots...")
        
        # Fallback to basic version
        try:
            from heuristic_tuner_simple import SimpleHeuristicTuner
            app = SimpleHeuristicTuner()
            app.run()
        except ImportError:
            print("Fallback version not available. Creating basic tuner...")
            create_basic_tuner()

def create_basic_tuner():
    """Create a basic tuner without matplotlib dependencies"""
    import tkinter as tk
    from tkinter import ttk
    
    class BasicTuner:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Basic Heuristic Controller Tuner")
            self.root.geometry("500x400")
            
            # Parameters
            self.params = {
                'base_margin': 6.0,
                'velocity_factor': 0.9,
                'approach_distance': 180,
            }
            
            self.setup_gui()
        
        def setup_gui(self):
            main_frame = ttk.Frame(self.root, padding=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            title = ttk.Label(main_frame, text="Heuristic Controller Parameters", 
                            font=("Arial", 14, "bold"))
            title.pack(pady=(0, 20))
            
            # Parameter controls
            for i, (param, value) in enumerate(self.params.items()):
                frame = ttk.Frame(main_frame)
                frame.pack(fill=tk.X, pady=5)
                
                label = ttk.Label(frame, text=param.replace('_', ' ').title() + ":")
                label.pack(side=tk.LEFT)
                
                var = tk.DoubleVar(value=value)
                scale = ttk.Scale(frame, from_=0, to=20 if 'margin' in param else 300,
                                variable=var, length=200)
                scale.pack(side=tk.RIGHT, padx=(10, 0))
                
                value_label = ttk.Label(frame, text=f"{value:.1f}")
                value_label.pack(side=tk.RIGHT)
                
                def update_value(val, p=param, lbl=value_label):
                    self.params[p] = float(val)
                    lbl.config(text=f"{float(val):.1f}")
                
                scale.config(command=update_value)
            
            # Explanation
            ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
            
            explanation = tk.Text(main_frame, height=8, wrap=tk.WORD)
            explanation.pack(fill=tk.BOTH, expand=True)
            
            explanation.insert(tk.END, """HEURISTIC CONTROLLER PARAMETERS:

• Base Margin: Distance below gap center before flapping
  - Lower values = more aggressive (higher scores, more risk)
  - Higher values = safer (lower scores, less crashes)

• Velocity Factor: How much falling speed affects decision
  - Higher = more reactive to falling
  - Lower = ignores speed, consistent behavior

• Approach Distance: When to start considering the pipe
  - Lower = flap later, more precise
  - Higher = flap earlier, safer

TUNING TIPS:
1. Start with base_margin (most important)
2. Adjust velocity_factor for speed response
3. Fine-tune approach_distance for timing""")
            
            explanation.config(state=tk.DISABLED)
            
            # Export button
            export_btn = ttk.Button(main_frame, text="Export Parameters", 
                                  command=self.export_params)
            export_btn.pack(pady=10)
        
        def export_params(self):
            print("Current parameters:")
            for param, value in self.params.items():
                print(f"  {param}: {value}")
        
        def run(self):
            self.root.mainloop()
    
    app = BasicTuner()
    app.run()

if __name__ == "__main__":
    main()
