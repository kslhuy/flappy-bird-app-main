"""
Neural Network Training Script for Flappy Bird Imitation Learning
Train a neural network to copy expert AI behavior from imitation_data.csv
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class FlappyBirdImitationNet(nn.Module):
    """Neural network for learning Flappy Bird gameplay from expert demonstrations"""
    
    def __init__(self, input_size=5, hidden_sizes=[64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)  # Regularization
            ])
            prev_size = hidden_size
        
        # Output layer for binary classification (flap/no-flap)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data(csv_path='imitation_data.csv'):
    """Load and preprocess the imitation learning dataset"""
    
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Action distribution:")
    print(data['action'].value_counts())
    
    # Features (game state)
    feature_columns = ['bird_y', 'bird_velocity', 'gap_top', 'gap_bottom', 'pipe_dx']
    X = data[feature_columns].values
    y = data['action'].values
    
    # Data quality checks
    print(f"\nData quality:")
    print(f"Missing values: {pd.isna(data).sum().sum()}")
    print(f"Feature ranges:")
    for col in feature_columns:
        print(f"  {col}: [{data[col].min():.2f}, {data[col].max():.2f}]")
    
    # Remove outliers (optional)
    # Filter out impossible states
    valid_mask = (
        (data['bird_y'] >= 0) & (data['bird_y'] <= 600) &
        (data['bird_velocity'] >= -15) & (data['bird_velocity'] <= 15) &
        (data['gap_top'] >= 0) & (data['gap_top'] <= 600) &
        (data['gap_bottom'] >= 0) & (data['gap_bottom'] <= 600) &
        (data['pipe_dx'] >= -50) & (data['pipe_dx'] <= 500)
    )
    
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"After filtering: {len(X)} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def train_model(X, y, test_size=0.2, epochs=100, learning_rate=0.001):
    """Train the neural network model"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create model
    model = FlappyBirdImitationNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    test_accuracies = []
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_predictions = model(X_train_tensor).squeeze()
        train_loss = criterion(train_predictions, y_train_tensor)
        
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # Evaluation (every 10 epochs)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test_tensor).squeeze()
                test_binary_pred = (test_predictions > 0.5).float()
                test_accuracy = (test_binary_pred == y_test_tensor).float().mean().item()
                test_accuracies.append(test_accuracy)
                
                print(f"Epoch {epoch:3d}: Loss = {train_loss.item():.4f}, "
                      f"Test Accuracy = {test_accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze()
        test_binary_pred = (test_predictions > 0.5).float()
        final_accuracy = (test_binary_pred == y_test_tensor).float().mean().item()
        
        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        
        # Confusion matrix
        tp = ((test_binary_pred == 1) & (y_test_tensor == 1)).sum().item()
        tn = ((test_binary_pred == 0) & (y_test_tensor == 0)).sum().item()
        fp = ((test_binary_pred == 1) & (y_test_tensor == 0)).sum().item()
        fn = ((test_binary_pred == 0) & (y_test_tensor == 1)).sum().item()
        
        print(f"Confusion Matrix:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"  Precision: {precision:.4f}")
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"  Recall: {recall:.4f}")
    
    return model, train_losses, test_accuracies

def save_model(model, scaler, model_path='flappy_bird_imitation_model.pth'):
    """Save the trained model and scaler"""
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_architecture': type(model).__name__
    }, model_path)
    
    print(f"\nModel saved to {model_path}")

def plot_training_history(train_losses, test_accuracies):
    """Plot training progress"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot test accuracy
    epochs_tested = list(range(0, len(train_losses), 10))
    ax2.plot(epochs_tested, test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training history saved to training_history.png")

def main():
    """Main training pipeline"""
    
    print("🤖 Flappy Bird Imitation Learning Training")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        X, y, scaler, feature_columns = load_and_preprocess_data()
        
        if len(X) < 100:
            print("⚠️  Warning: Very small dataset. Consider collecting more data.")
        
        # Train model
        model, train_losses, test_accuracies = train_model(X, y)
        
        # Save model
        save_model(model, scaler)
        
        # Plot results
        try:
            plot_training_history(train_losses, test_accuracies)
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
        
        print("\n✅ Training completed successfully!")
        print("\nNext steps:")
        print("1. Add ImitationController to ai_controllers.py")
        print("2. Test the trained model in the game")
        print("3. Compare performance with other controllers")
        
    except FileNotFoundError:
        print("❌ Error: imitation_data.csv not found!")
        print("Please run the game with AI enabled to collect training data first.")
    
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()
