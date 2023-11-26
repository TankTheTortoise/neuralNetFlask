
from PIL import Image, ImageChops
import numpy as np
import pickle

from flask_wtf.file import *
from wtforms import SubmitField

from MyNeuralNet.data_utils import *
from fileinput import filename
from flask import *
import os
from scripts.user_utils import guess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/images"

@app.route('/')
def main():
    return render_template("index.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(app.config['UPLOAD_FOLDER']+"/"+f.filename)
        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        num = guess(full_filepath)
        return render_template("Acknowledgement.html", name=f.filename, full_filepath=full_filepath, num=num)










    # file_path = "/home/tortoise/PycharmProjects/ActiveNeuralNetwork/images/one.jpg"
    # image = Image.open(file_path).convert('L')
    # image = image.resize((28, 28))
    # image = ImageChops.invert(image)
    #
    # image = np.array(image)
    # image = image / 255
    # image = image.reshape(1, -1)
    # with open('/home/tortoise/PycharmProjects/ActiveNeuralNetwork/models/model.pickle', 'rb') as f:
    #     hello = pickle.load(f)
