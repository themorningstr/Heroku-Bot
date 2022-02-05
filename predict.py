import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


class DiseaseClassification:
    def __init__(self, filename):
        self.filename = filename

    def predictionDisease(self):
        # Load Model

        model = load_model("Model/model_inception_epoch_10.h5")

        imageName = self.filename
        testImage = image.load_img(imageName, target_size = (512,512))
        testImage = image.img_to_array(testImage)
        testImage = testImage / 255
        testImage = np.expand_dims(testImage, axis = 0)
        testImage = preprocess_input(testImage)
        result = model.predict(testImage)
        classLabel = np.argmax(result, axis = 1)
        return result, classLabel

    def className(self, result):
        if result == 0:
            prediction = "Healthy"
        elif result == 1:
            prediction = "Leaf Rust"
        elif result == 2:
            prediciton = "Powdery Mildew"
        elif result == 3:
            prediction = "Septoria"
        else:
            prediction = "Yellow Rust"

        return prediction
        

        


