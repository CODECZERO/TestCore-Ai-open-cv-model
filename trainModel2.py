import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ================================
# Step 1: Load and Preprocess Data
# ================================

def load_data():
    """ Load and preprocess dataset from CSV with noise to prevent overfitting """
    data = pd.read_csv("eye_tracking_data.csv")

    # Convert text labels into numbers (0 = Normal, 1 = Cheating)
    def map_label(label):
        if pd.isnull(label) or str(label).strip() == "" or label.strip() == "Looking Center":
            return 0
        else:
            return 1

    data["LabelNum"] = data["Label"].apply(map_label)
    data = data.dropna(subset=["LeftEAR", "RightEAR", "VerticalDiff", "LabelNum"])

    # Extract features (X) and labels (y)
    X = data[["LeftEAR", "RightEAR", "VerticalDiff"]].values.astype(np.float32)
    y = data["LabelNum"].values.astype(np.float32)

    # Add slight noise for better generalization (prevents memorization)
    noise = np.random.normal(0, 0.005, X.shape)  # Small noise to EAR values
    X += noise  

    return X, y

# ================================
# Step 2: Define or Load Model
# ================================

def create_model():
    """ Create a new neural network model with learning rate decay """
    model = Sequential([
        Dense(16, activation='relu', input_shape=(3,)),  
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Output probability of cheating
    ])
    
    optimizer = Adam(learning_rate=0.01)  # Start with a high learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_or_create_model(model_path="eye_tracking_model_final.keras"):
    """ Load existing model if available, else create a new one """
    if os.path.exists(model_path):
        print("🔄 Loading existing model...")
        return load_model(model_path)
    else:
        print("🆕 No existing model found. Creating a new one...")
        return create_model()

# ================================
# Step 3: Train the Model Repeatedly
# ================================

def train_model(iterations=10, model_path="eye_tracking_model_final.keras"):
    """ Train the model multiple times, improving it with each iteration """
    X, y = load_data()  # Load dataset
    model = load_or_create_model(model_path)  # Load or create model
    best_accuracy = 0.0  # Store the best accuracy

    # Track important data for graphs
    history_data = {
        "accuracy": [],
        "loss": [],
        "learning_rate": []
    }

    for i in range(iterations):
        print(f"\n🔄 **Training Iteration {i+1}/{iterations}**")

        # Reduce learning rate after each iteration
        new_lr = 0.01 / (1 + 0.1 * i)  # Decreasing learning rate
        model.optimizer.learning_rate.assign(new_lr)
        print(f"🔽 Adjusted Learning Rate: {new_lr:.6f}")

        # Train the model
        history = model.fit(X, y, epochs=10, batch_size=8, verbose=1)

        # Evaluate model performance
        loss, accuracy = model.evaluate(X, y, verbose=0)
        accuracy_percentage = accuracy * 100
        print(f"📊 Accuracy after iteration {i+1}: {accuracy_percentage:.2f}%")

        # Save only if it's the best model so far
        if accuracy_percentage > best_accuracy:
            best_accuracy = accuracy_percentage
            model.save(model_path)
            print(f"🚀 New Best Model Saved! Accuracy: {accuracy_percentage:.2f}%")
        else:
            print(f"🔁 Model did not improve. Best Accuracy: {best_accuracy:.2f}%")

        # Store data for graph plotting
        history_data["accuracy"].append(accuracy_percentage)
        history_data["loss"].append(loss)
        history_data["learning_rate"].append(new_lr)

        # Stop training if we reach 99% accuracy
        if accuracy_percentage > 99:
            print("🎉 Model has reached high accuracy! Stopping training.")
            break

        time.sleep(2)  # Pause before next iteration

    # Plot training progress
    plot_training_progress(history_data)

    return model

# ================================
# Step 4: Plot Training Progress
# ================================

def plot_training_progress(history_data):
    """ Plot loss, accuracy, and learning rate over iterations """
    plt.figure(figsize=(15, 5))

    # Accuracy Plot
    plt.subplot(1, 3, 1)
    plt.plot(history_data["accuracy"], label='Training Accuracy', marker='o', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Iterations')
    plt.ylim(0, 100)  # Accuracy is percentage-based
    plt.legend()

    # Loss Plot
    plt.subplot(1, 3, 2)
    plt.plot(history_data["loss"], label='Training Loss', marker='o', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()

    # Learning Rate Plot
    plt.subplot(1, 3, 3)
    plt.plot(history_data["learning_rate"], label='Learning Rate', marker='o', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ================================
# Step 5: Run Training
# ================================
if __name__ == "__main__":
    trained_model = train_model(iterations=10)
