'''from flask import Flask, render_template, request
from utils import predict_image, highlight_psoriasis
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)

            output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
            highlight_psoriasis(filepath, output_path)

            return render_template('result.html',
                                   label=label,
                                   confidence=confidence,
                                   original_img=filepath,
                                   output_img=output_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, request, jsonify
from utils import predict_image, highlight_psoriasis
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        label, confidence = predict_image(filepath)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
        highlight_psoriasis(filepath, output_path)

        return jsonify({
            "label": label,
            "confidence": float(confidence),  
            "original_img": filepath,
            "output_img": output_path
        })


if __name__ == '__main__':
    app.run(debug=True)

from flask import send_from_directory

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


