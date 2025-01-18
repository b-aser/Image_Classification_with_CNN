from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

# Define CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((32, 32))  # Resize to CIFAR-10 input size
        img = np.array(img) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Return the prediction as a JSON response
        return jsonify({
            'prediction': CIFAR10_CLASSES[predicted_class],
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
