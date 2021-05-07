from keras.preprocessing import image
from keras import models
import numpy as np
from PIL import Image


model = models.load_model('./model')

def load_image(caminho):
  return Image.open(caminho).convert('RGB')

def formatImage(image):
  image = image.resize((64,64), Image.LINEAR)
  image = image.img_to_array(image)
  image /= 255
  image = np.expand_dims(image, axis = 0)
  return image

normal_image = load_image('chest_xray_dataset/test/NORMAL/IM-0001-0001.jpeg')

image_pneumonia = load_image('chest_xray_dataset/test/PNEUMONIA/person1_virus_6.jpeg')

normal_image = formatImage(normal_image)
image_pneumonia = formatImage(image_pneumonia)

normal_prediction = model.predict(normal_image)
pneumonia_prediction = model.predict(image_pneumonia)
print(normal_prediction)
print(pneumonia_prediction)
