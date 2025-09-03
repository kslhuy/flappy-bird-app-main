# Imitation Learning in Flappy Bird AI

## 🤖 What is Imitation Learning?

**Imitation Learning** (also called **Behavioral Cloning**) is a machine learning approach where an AI learns by copying the behavior of an expert. Instead of learning through trial and error (like reinforcement learning), the AI observes demonstrations and tries to mimic them.

## 📊 The `imitation_data.csv` File

### Purpose
The CSV file captures **state-action pairs** from expert gameplay. Each row represents:
- **State**: Current game situation (bird position, velocity, pipe positions)
- **Action**: What the expert did in that situation (0 = no flap, 1 = flap)

### Data Format
```csv
bird_y,bird_velocity,gap_top,gap_bottom,pipe_dx,action
300,0,169,319,300,0
301,1.0,169,319,294,0
360,7.5,169,319,252,0
376,8.5,169,319,249,1  # Expert decided to flap here!
```

### Data Fields Explained:
- **bird_y**: Bird's vertical position (0 = top, 600 = bottom)
- **bird_velocity**: Bird's falling/rising speed
- **gap_top**: Top edge of the pipe gap
- **gap_bottom**: Bottom edge of the pipe gap  
- **pipe_dx**: Horizontal distance to the pipe
- **action**: Expert's decision (0 = don't flap, 1 = flap)

## 🎯 How Does Data Collection Work?

### Current Implementation
In your current system, **ALL AI controllers** log their decisions:

```python
def ai_decide():
    # ... controller logic ...
    logger.log_sample(state, action)  # Logs every AI decision
    return action
```

### Who Generates the Data?
1. **Heuristic Controller**: Rule-based expert decisions
2. **PID Controller**: Mathematical control decisions  
3. **Planner Controller**: Physics simulation decisions
4. **Human Player**: Manual gameplay (when AI is off)

## 🧠 Training a Neural Network Bird

### Step 1: Collect Expert Data
```python
# Run the game with a good AI controller (like Heuristic)
# It automatically saves decisions to imitation_data.csv
python flappybird.py
```

### Step 2: Train a Neural Network
Here's how you could train a neural network to copy the expert:

```python
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class FlappyBirdNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),      # 5 input features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),      # Output: probability to flap
            nn.Sigmoid()           # 0-1 probability
        )
    
    def forward(self, x):
        return self.network(x)

# Load and prepare data
data = pd.read_csv('imitation_data.csv')
X = data[['bird_y', 'bird_velocity', 'gap_top', 'gap_bottom', 'pipe_dx']].values
y = data['action'].values

# Normalize inputs
X_normalized = X / [600, 15, 600, 600, 400]  # Scale to 0-1 range

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2)

# Create and train model
model = FlappyBirdNet()
criterion = nn.BCELoss()  # Binary classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(torch.FloatTensor(X_train))
    loss = criterion(predictions.squeeze(), torch.FloatTensor(y_train))
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Save trained model
torch.save(model.state_dict(), 'flappy_bird_imitation_model.pth')
```

### Step 3: Create Neural Network Controller
```python
class NeuralController:
    def __init__(self, model_path):
        self.model = FlappyBirdNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def decide(self, state, bird_height):
        # Normalize inputs
        inputs = torch.FloatTensor([
            state['bird_y'] / 600,
            state['bird_velocity'] / 15,
            state['gap_top'] / 600,
            state['gap_bottom'] / 600,
            state['pipe_dx'] / 400
        ])
        
        with torch.no_grad():
            flap_probability = self.model(inputs).item()
        
        # Flap if probability > 0.5
        return 1 if flap_probability > 0.5 else 0
```

## 🎮 Complete Training Pipeline

### Phase 1: Expert Data Collection
1. Run game with **Heuristic Controller** (best performing)
2. Let it play for thousands of games
3. Collect high-quality state-action pairs

### Phase 2: Data Preprocessing
```python
# Remove poor examples (games that crashed quickly)
# Balance the dataset (equal flap/no-flap examples)
# Add data augmentation (slight position variations)
```

### Phase 3: Neural Network Training
```python
# Train supervised learning model
# Use techniques like:
# - Cross-validation
# - Regularization (dropout, weight decay)
# - Learning rate scheduling
```

### Phase 4: Evaluation & Improvement
```python
# Test neural network performance
# Compare with original expert
# Iterate and improve
```

## 🔄 Advanced Techniques

### 1. DAgger (Dataset Aggregation)
Instead of just learning from initial expert data:
1. Train initial model on expert data
2. Let model play and collect its decisions
3. Ask expert to correct model's mistakes
4. Retrain on combined original + corrected data
5. Repeat until performance improves

### 2. Multi-Expert Learning
Combine data from multiple experts:
- Heuristic Controller (rule-based)
- PID Controller (mathematical)
- Human players (intuitive)

### 3. Residual Learning
Your code already has this! The `ResidualNet` learns corrections to the heuristic:
```python
# Heuristic makes base decision
margin = base_margin + velocity_factor * bird_velocity

# Neural network adds refinement
residual = residual_model(state_features) * 10
margin += residual
```

## 🚀 Why Use Imitation Learning?

### Advantages:
1. **No Reward Engineering**: Don't need to design reward functions
2. **Fast Learning**: Learn from expert demonstrations directly
3. **Safe Learning**: No exploration in dangerous states
4. **Human-Like Behavior**: Can capture human playing style

### Disadvantages:
1. **Limited by Expert**: Can't exceed expert performance
2. **Requires Expert Data**: Need good demonstrations
3. **Distribution Shift**: May fail in unseen situations
4. **No Improvement**: Doesn't learn from its own mistakes

## 🔧 How to Use in Your Project

### Current Status:
Your game is **already collecting imitation data** from all AI controllers! Every time an AI makes a decision, it's logged.

### To Train Your Own Neural Bird:
1. **Collect Data**: Run the game with Heuristic controller for 30+ minutes
2. **Analyze Data**: Check `imitation_data.csv` for quality patterns
3. **Train Model**: Use the code examples above
4. **Add Controller**: Integrate trained neural network as new AI mode
5. **Compare Performance**: Test against original controllers

### Example Integration:
```python
# In ai_controllers.py
class ImitationController:
    def __init__(self, model_path):
        self.model = load_trained_model(model_path)
    
    def decide(self, state, bird_height):
        return self.model.predict(state)

# In create_controller function:
elif mode == 'imitation':
    return ImitationController('trained_model.pth')
```

## 🎯 Summary

Imitation learning turns your Flappy Bird into a **learning system** where:
1. **Experts demonstrate** good gameplay
2. **Neural networks learn** to copy expert decisions  
3. **New AI emerges** that plays like the expert
4. **Performance can be measured** against original expert

The `imitation_data.csv` is the **bridge** between expert knowledge and AI learning - it's the training dataset that teaches machines to play like experts!
