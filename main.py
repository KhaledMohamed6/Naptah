from flask import Flask, request
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from PIL import Image
import torch

app = Flask(__name__)

model = AutoModelForImageClassification.from_pretrained(
    '../apiModel/myModel')
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k")


@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Get the image file from the request
    image_file = request.files['image']

    # Save the image file to a desired location on the server
    image_path = "assets/img.jpg"
    image_file.save(image_path)

    # You can perform additional operations with the image here
    # ...

    return 'Image uploaded successfully'


@app.route('/get_text', methods=['GET'])
def get_text():
    image = Image.open('assets/img.jpg')
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    disease = model.config.id2label[predicted_label]

    return disease


if __name__ == '__main__':
    app.run(host='192.168.1.7', port=5000)
