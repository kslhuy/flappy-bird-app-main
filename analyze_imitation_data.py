"""
Quick Demo: Analyze Your Existing Imitation Data
This script analyzes the imitation_data.csv to show you what the AI has learned
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_imitation_data():
    """Analyze the collected imitation learning data"""
    
    print("🔍 Analyzing Imitation Learning Data")
    print("=" * 50)
    
    try:
        # Load data
        data = pd.read_csv('imitation_data.csv')
        print(f"📊 Total samples: {len(data):,}")
        
        # Basic statistics
        print(f"\n📈 Action Distribution:")
        action_counts = data['action'].value_counts()
        for action, count in action_counts.items():
            action_name = "FLAP" if action == 1 else "NO FLAP"
            percentage = (count / len(data)) * 100
            print(f"  {action_name}: {count:,} ({percentage:.1f}%)")
        
        # Analyze when the AI decides to flap
        flap_data = data[data['action'] == 1]
        no_flap_data = data[data['action'] == 0]
        
        print(f"\n🦅 When does the AI FLAP?")
        if len(flap_data) > 0:
            print(f"  Average bird position: {flap_data['bird_y'].mean():.1f}")
            print(f"  Average bird velocity: {flap_data['bird_velocity'].mean():.2f}")
            print(f"  Average distance to pipe: {flap_data['pipe_dx'].mean():.1f}")
            
            # Gap center when flapping
            flap_data['gap_center'] = (flap_data['gap_top'] + flap_data['gap_bottom']) / 2
            flap_data['distance_from_gap_center'] = flap_data['bird_y'] - flap_data['gap_center']
            print(f"  Average distance from gap center: {flap_data['distance_from_gap_center'].mean():.1f}")
        
        print(f"\n🚫 When does the AI NOT FLAP?")
        if len(no_flap_data) > 0:
            print(f"  Average bird position: {no_flap_data['bird_y'].mean():.1f}")
            print(f"  Average bird velocity: {no_flap_data['bird_velocity'].mean():.2f}")
            print(f"  Average distance to pipe: {no_flap_data['pipe_dx'].mean():.1f}")
            
            # Gap center when not flapping
            no_flap_data['gap_center'] = (no_flap_data['gap_top'] + no_flap_data['gap_bottom']) / 2
            no_flap_data['distance_from_gap_center'] = no_flap_data['bird_y'] - no_flap_data['gap_center']
            print(f"  Average distance from gap center: {no_flap_data['distance_from_gap_center'].mean():.1f}")
        
        # Expert strategy analysis
        print(f"\n🧠 Expert Strategy Analysis:")
        
        # When approaching pipes
        approaching_pipe = data[data['pipe_dx'] < 100]
        if len(approaching_pipe) > 0:
            flap_rate_approaching = (approaching_pipe['action'] == 1).mean()
            print(f"  Flap rate when approaching pipe: {flap_rate_approaching:.1%}")
        
        # When falling vs rising
        falling = data[data['bird_velocity'] > 0]
        rising = data[data['bird_velocity'] < 0]
        
        if len(falling) > 0:
            flap_rate_falling = (falling['action'] == 1).mean()
            print(f"  Flap rate when falling: {flap_rate_falling:.1%}")
        
        if len(rising) > 0:
            flap_rate_rising = (rising['action'] == 1).mean()
            print(f"  Flap rate when rising: {flap_rate_rising:.1%}")
        
        # Position relative to gap
        data['gap_center'] = (data['gap_top'] + data['gap_bottom']) / 2
        data['relative_position'] = data['bird_y'] - data['gap_center']
        
        above_gap = data[data['relative_position'] < -20]  # 20 pixels above gap center
        below_gap = data[data['relative_position'] > 20]   # 20 pixels below gap center
        
        if len(above_gap) > 0:
            flap_rate_above = (above_gap['action'] == 1).mean()
            print(f"  Flap rate when above gap center: {flap_rate_above:.1%}")
        
        if len(below_gap) > 0:
            flap_rate_below = (below_gap['action'] == 1).mean()
            print(f"  Flap rate when below gap center: {flap_rate_below:.1%}")
        
        # Data quality check
        print(f"\n📋 Data Quality:")
        print(f"  Complete cases: {len(data.dropna()):,}")
        print(f"  Missing values: {data.isnull().sum().sum()}")
        
        # Feature ranges
        print(f"  Bird Y range: [{data['bird_y'].min():.0f}, {data['bird_y'].max():.0f}]")
        print(f"  Velocity range: [{data['bird_velocity'].min():.2f}, {data['bird_velocity'].max():.2f}]")
        print(f"  Pipe distance range: [{data['pipe_dx'].min():.0f}, {data['pipe_dx'].max():.0f}]")
        
        # Simple patterns
        print(f"\n🎯 Simple Patterns Discovered:")
        
        # Pattern 1: Emergency flaps (high velocity downward)
        emergency_situations = data[data['bird_velocity'] > 8]
        if len(emergency_situations) > 0:
            emergency_flap_rate = (emergency_situations['action'] == 1).mean()
            print(f"  Emergency flap rate (velocity > 8): {emergency_flap_rate:.1%}")
        
        # Pattern 2: Preemptive flaps (below gap center, approaching pipe)
        preemptive = data[(data['relative_position'] > 10) & (data['pipe_dx'] < 150)]
        if len(preemptive) > 0:
            preemptive_flap_rate = (preemptive['action'] == 1).mean()
            print(f"  Preemptive flap rate (below gap, approaching): {preemptive_flap_rate:.1%}")
        
        # Save analysis
        try:
            create_visualization(data)
        except Exception as e:
            print(f"  (Visualization skipped: {e})")
        
        print(f"\n✅ Analysis complete! Your AI has learned:")
        print(f"   - When to flap vs when to coast")
        print(f"   - Emergency maneuvers for high-speed situations")
        print(f"   - Position-based decision making")
        print(f"   - Distance-aware pipe navigation")
        
        print(f"\n💡 To train a neural network:")
        print(f"   python train_imitation_model.py")
        
    except FileNotFoundError:
        print("❌ imitation_data.csv not found!")
        print("Run the game with AI enabled to collect data first.")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

def create_visualization(data):
    """Create simple visualizations of the data"""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Action distribution
        action_counts = data['action'].value_counts()
        axes[0, 0].bar(['No Flap', 'Flap'], action_counts.values)
        axes[0, 0].set_title('Action Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Bird position when flapping vs not flapping
        flap_positions = data[data['action'] == 1]['bird_y']
        no_flap_positions = data[data['action'] == 0]['bird_y']
        
        axes[0, 1].hist([no_flap_positions, flap_positions], bins=30, alpha=0.7, 
                       label=['No Flap', 'Flap'], color=['blue', 'red'])
        axes[0, 1].set_title('Bird Position Distribution')
        axes[0, 1].set_xlabel('Bird Y Position')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Plot 3: Velocity when flapping vs not flapping
        flap_velocities = data[data['action'] == 1]['bird_velocity']
        no_flap_velocities = data[data['action'] == 0]['bird_velocity']
        
        axes[1, 0].hist([no_flap_velocities, flap_velocities], bins=30, alpha=0.7,
                       label=['No Flap', 'Flap'], color=['blue', 'red'])
        axes[1, 0].set_title('Bird Velocity Distribution')
        axes[1, 0].set_xlabel('Bird Velocity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Plot 4: Distance to pipe when flapping
        flap_distances = data[data['action'] == 1]['pipe_dx']
        no_flap_distances = data[data['action'] == 0]['pipe_dx']
        
        axes[1, 1].hist([no_flap_distances, flap_distances], bins=30, alpha=0.7,
                       label=['No Flap', 'Flap'], color=['blue', 'red'])
        axes[1, 1].set_title('Distance to Pipe Distribution')
        axes[1, 1].set_xlabel('Distance to Pipe')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('imitation_data_analysis.png', dpi=150, bbox_inches='tight')
        print(f"  📊 Visualization saved to imitation_data_analysis.png")
        
    except ImportError:
        print("  📊 Install matplotlib for visualizations: pip install matplotlib")

if __name__ == "__main__":
    analyze_imitation_data()
