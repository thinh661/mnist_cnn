import os
import numpy as np
import keras
from keras.models import load_model
import cv2
from PIL import Image

model = load_model('model_v2.h5')
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

folder_path = r"D:\WorkSpace_Thinh1\CNN_ML_Project\data_real"

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(folder_path, filename)
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        im = Image.fromarray(frame)
        im = im.resize((28, 28))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=2)
        img_array = np.expand_dims(img_array, axis=0)

        predict = model.predict(1.0 - img_array)
        result = digits[np.argmax(predict)]
        print(f"File: {filename}, Predicted Digit: {result}")
    else:
        continue
