import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pandas as pd
from PIL import Image

# Paths
test_dir = r"C:\Users\Acer\Documents\LargeDS\LargeDS\test"
checkpoint_dir = r"C:\Users\Acer\Documents\LargeDS\LargeDS"
model_path = os.path.join(checkpoint_dir, 'final_model_resnet50v2.keras')  # or 'best_model_resnet50v2.keras'
class_indices_path = os.path.join(checkpoint_dir, 'class_indices.json')

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 224, 224, 16
model_name = 'resnet50v2'

# Validate test directory
if not os.path.exists(test_dir):
    print(f"Error: Test directory {test_dir} does not exist.")
    exit(1)
if not os.path.isdir(test_dir):
    print(f"Error: {test_dir} is not a directory.")
    exit(1)
class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
if not class_dirs:
    print(f"Error: No class subdirectories found in {test_dir}.")
    exit(1)
print(f"Found {len(class_dirs)} class subdirectories in {test_dir}: {class_dirs}")

# Validate images
total_images = 0
for class_dir in class_dirs:
    class_path = os.path.join(test_dir, class_dir)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in images:
        try:
            img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
            if img.size != (IMG_HEIGHT, IMG_WIDTH):
                img = img.resize((IMG_HEIGHT, IMG_WIDTH))
            img.close()
            total_images += 1
        except Exception as e:
            print(f"Warning: Invalid image {os.path.join(class_path, img_file)}: {e}")
    print(f"Found {len(images)} images in {class_path}")
print(f"Total valid images: {total_images}")

# Load class indices
if not os.path.exists(class_indices_path):
    print(f"Error: Class indices file not found at {class_indices_path}")
    exit(1)
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
num_classes = len(class_indices)
class_labels = list(class_indices.keys())
print(f"Loaded {num_classes} classes: {class_labels}")

# Verify class match
if set(class_dirs) != set(class_indices.keys()):
    print("Error: Test directory classes differ from class_indices.json.")
    print(f"Test classes: {class_dirs}")
    print(f"Saved classes: {list(class_indices.keys())}")
    exit(1)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
try:
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
except ValueError as e:
    print(f"Error in test generator: {e}")
    print(f"Test directory: {test_dir}")
    print(f"Subdirectories: {class_dirs}")
    print(f"Total images: {total_images}")
    print("Ensure all images are valid RGB images and subdirectories match class_indices.json.")
    exit(1)

# Verify generator output
print(f"Test generator classes: {list(test_generator.class_indices.keys())}")
print(f"Test samples: {test_generator.n}, Expected classes: {num_classes}")

# Load the saved model
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'checkpoint_{model_name}_epoch_') and f.endswith('.keras')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.keras')[0]))
        model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Falling back to latest checkpoint: {model_path}")
    else:
        print("No checkpoints found. Please train the model first.")
        exit(1)
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# Calculate test steps
test_steps = max(1, (test_generator.n + BATCH_SIZE - 1) // BATCH_SIZE)
print(f"Test samples: {test_generator.n}, Test steps: {test_steps}")

# Evaluate on test set
try:
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps, verbose=1)
    print(f"Overall Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)

# Generate predictions
test_generator.reset()
try:
    predictions = model.predict(test_generator, steps=test_steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
except Exception as e:
    print(f"Error during prediction: {e}")
    exit(1)

# Handle length mismatch
min_samples = min(len(true_classes), len(predicted_classes))
if len(true_classes) != len(predicted_classes):
    print(f"Warning: Mismatch in true_classes ({len(true_classes)}) and predicted_classes ({len(predicted_classes)})")
true_classes = true_classes[:min_samples]
predicted_classes = predicted_classes[:min_samples]

# Ensure class labels match
unique_classes = np.unique(true_classes)
class_labels = [label for label in class_labels if class_indices[label] in unique_classes]
if len(class_labels) != num_classes:
    print(f"Warning: Number of classes ({len(class_labels)}) does not match expected ({num_classes})")
    class_labels = class_labels[:num_classes]

# Classification report
report = classification_report(true_classes, predicted_classes, labels=range(num_classes), target_names=class_labels, zero_division=0, output_dict=True)
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, labels=range(num_classes), target_names=class_labels, zero_division=0))

# Save classification report to CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(checkpoint_dir, f'classification_report_{model_name}.csv'))
print(f"Classification report saved to {checkpoint_dir}/classification_report_{model_name}.csv")

# Classification report heatmap
report_df_metrics = report_df[['precision', 'recall', 'f1-score']].iloc[:-3]  # Exclude accuracy, macro avg, weighted avg
plt.figure(figsize=(12, 8))
sns.heatmap(report_df_metrics, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.title('Classification Report: Precision, Recall, F1-Score')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, f'classification_report_heatmap_{model_name}.png'))
plt.close()
print(f"Classification report heatmap saved to {checkpoint_dir}/classification_report_heatmap_{model_name}.png")

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes, labels=range(num_classes))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, f'confusion_matrix_{model_name}.png'))
plt.close()
print(f"Confusion matrix saved to {checkpoint_dir}/confusion_matrix_{model_name}.png")