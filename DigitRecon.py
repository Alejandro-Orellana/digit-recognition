from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# load trained model
model = tf.keras.models.load_model('mnist_model.keras')

@app.route('/', methods=['GET'])
def index():
	# Render the index page
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	file = request.files['file']
	image_bytes = file.read()
	image = Image.open(io.BytesIO(image_bytes))

	# Preprocess the image
	image = image.resize((28, 28)).convert('L')
	image = np.array(image)
	image = image / 255.0
	image = image.reshape(1, 28, 28, 1) # Reshare for model

	# Make a prediction
	prediction = model.predict(image)
	digit = np.argmax(prediction)

	return f'Predicted Digit: {digit}'

if __name__ == '__main__':
	app.run(debug=True)