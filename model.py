from keras.preprocessing import image
from keras import models
import numpy as np
from PIL import Image

class Model:
  def __init__(self, modelName:str):
    self.model : models.Model = models.load_model(modelName)

  def __carregar_image(self, caminho):
    return Image.open(caminho).convert('RGB')

  def __formatar_image(self, image):
    image = image.resize((64,64), Image.LINEAR)
    image = image.img_to_array(image)
    image = image/255
    image = np.expand_dims(image, axis = 0)
    return image

  def load_weights(self, filename):
    self.model.load_weights(filename)

  def predict_image_from_path(self, imagePath: str):
    image = self.__carregar_image(imagePath)
    image = self.__formatar_image(image)
    prediction = self.model.predict(image)
    if prediction>0.7:
      return 'Pneumonia'
    return 'Normal' 