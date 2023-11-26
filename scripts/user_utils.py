import pickle
import numpy as np
from PIL import Image, ImageChops


def guess(file_path):
    image = Image.open(file_path).convert('L')
    image = image.resize((28, 28))
    image = ImageChops.invert(image)

    image = np.array(image)
    image = image / 255
    image = image.reshape(1, -1)

    with open('models/model.pickle', 'rb') as f:
        hello = pickle.load(f)
    return np.argmax(hello.predict(image))
