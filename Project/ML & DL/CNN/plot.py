import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
model = tf.keras.models.load_model('cnn_hw_nas_full_model2.keras')

# Function to predict and show results
def predict_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Adjust size if needed
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Show image and result
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence

# Use it
predicted_class, confidence = predict_image('your_image.jpg')
print(f"Class: {predicted_class}, Confidence: {confidence:.2%}")




