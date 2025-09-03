from cothread import catools
from softioc import builder
import tensorflow as tf
from tensorflow import keras
import cv2
from io import BytesIO
import urllib
from time import sleep
import numpy as np

try:
    import epicscorelibs.path.cothread
except ImportError:
    pass


class GoniOwlController:
    def __init__(self):
        self.model = tf.keras.models.load_model("./categorical_11072024.h5")
        self.model.summary()
        self.classes = ["dark", "light", "pinoff", "pinon"]

    def urltoimage(self, url):
        self.resp = urllib.request.urlopen(url)
        self.image = np.asarray(bytearray(self.resp.read()), dtype="uint8")
        self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
        self.image = self.image[100:900, 100:900]
#        self.image = self.image[400:700, 610:770]
        cv2.imwrite("tmp.jpg", self.image)
        _, buffer = cv2.imencode(".jpg", self.image)
        io_buf = BytesIO(buffer)
        return io_buf

    def infer(self):
        self.stream = self.urltoimage(
            "http://bl23i-di-serv-04.diamond.ac.uk:8080/ECAM6.mjpg.jpg"
        )
        self.img_in = keras.preprocessing.image.load_img(
            (self.stream), target_size=(800, 800)
        )
        self.img_array = keras.preprocessing.image.img_to_array(self.img_in)
        self.img_array = tf.expand_dims(self.img_array, 0)
        self.predictions = self.model.predict(self.img_array, verbose=0)
        self.score = tf.nn.softmax(self.predictions[0])
        print(np.argmax(self.score))
        # print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(classes[np.argmax(score)], 100 * np.max(score)))
        print(
            "Status is probably {} with {:.2f} % conf.".format(
                self.classes[np.argmax(self.score)], 200 * np.max(self.score)
            )
        )
        return np.argmax(self.score)
