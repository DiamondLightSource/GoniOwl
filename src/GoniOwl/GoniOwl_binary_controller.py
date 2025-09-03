from cothread import catools
from softioc import builder
import tensorflow as tf
from tensorflow import keras
import cv2
from io import BytesIO
import urllib
from time import sleep
import numpy as np
import datetime


try:
    import epicscorelibs.path.cothread
except ImportError:
    pass

# 20250714-111015_epoch20_binary_batch4.h5 is 1/5 scaled image! 

image_divider = 5

class GoniOwlController:
    def __init__(self):
        self.log_path = "/dls/science/groups/i23/scripts/chris/GoniOwl/src/GoniOwl/logs/GoniOwl_binary_controller.log"
        with open(self.log_path, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - INFO - GoniOwl binary controller started.\n")
        self.model_name = "20250714-111015_epoch20_binary_batch4.h5"
        self.model = tf.keras.models.load_model(f"./{self.model_name}")
        with open(self.log_path, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - INFO - Model {self.model_name} loaded successfully.\n")
        self.model.summary()
        self.classes = ["pinoff", "pinon"]


    def urltoimage(self, url):
        self.resp = urllib.request.urlopen(url)
        self.image = np.asarray(bytearray(self.resp.read()), dtype="uint8")
        self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
        self.image = self.image[100:-100, 100:-100]
        cv2.imwrite("tmp.jpg", self.image)
        _, buffer = cv2.imencode(".jpg", self.image)
        io_buf = BytesIO(buffer)
        return io_buf

    def infer(self):
        self.stream = self.urltoimage(
            "http://bl23i-di-serv-04.diamond.ac.uk:8080/ECAM6.mjpg.jpg"
        )
        self.stream.seek(0)
        img_bytes = self.stream.read()
        img_tensor = tf.image.decode_image(img_bytes, channels=3)
        img_height, img_width = int(img_tensor.shape[0]), int(img_tensor.shape[1])
        self.stream.seek(0)
        self.img_in = keras.preprocessing.image.load_img(
            (self.stream), target_size=(int(img_height / image_divider), int(img_width / image_divider))
        )
        self.img_array = keras.preprocessing.image.img_to_array(self.img_in)
        self.img_array = tf.expand_dims(self.img_array, 0)
        self.predictions = self.model.predict(self.img_array, verbose=0)
        self.score = self.predictions[0]

        if self.score < 0.15:
            print(f"{self.score} is {self.classes[0]}")
            self.predicted_label = self.classes[0]
            self.score = 1 - self.score
            self.printscore()
            return 1
        elif self.score > 0.85:
            print(f"{self.score} is {self.classes[1]}")
            self.predicted_label = self.classes[1]
            self.printscore()
            return 2
        else:
            print(f"{self.score} is unknown")
            self.predicted_label = "unknown"
            self.printscore()
            return 3
        
    def printscore(self):
        rounded_score = np.round((self.score * 100), 3)
        msg = f"Status is {self.predicted_label} with {str(rounded_score)} % conf."
        print(msg)
        
        with open(self.log_path, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - INFO - {msg}\n")