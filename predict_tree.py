import tensorflow as tf
import numpy as np
from PIL import Image
import json

model = tf.keras.models.load_model(r'C:\Users\Acer\Documents\LargeDS\LargeDS\best_model_resnet50v2.keras')
with open(r'C:\Users\Acer\Documents\LargeDS\LargeDS\class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

img_path = r"C:\Users\Acer\Documents\LargeDS\LargeDS\Vathakkani\Vathakkani_109.png"
img = Image.open(img_path).convert('RGB').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]
confidence = np.max(prediction)
print(f"Predicted species: {class_labels[predicted_class]} (Confidence: {confidence:.4f})")