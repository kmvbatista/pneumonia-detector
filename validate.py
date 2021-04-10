from keras.preprocessing import image
from keras import models
import numpy as np
from PIL import Image


model = models.load_model('./modelo')

def carregar_imagem(caminho):
  return Image.open(caminho).convert('RGB')

def formatar_imagem(imagem):
  imagem = imagem.resize((64,64), Image.LINEAR)
  imagem = image.img_to_array(imagem)
  imagem /= 255
  imagem = np.expand_dims(imagem, axis = 0)
  return imagem

imagem_normal = carregar_imagem('chest_xray_dataset/train/NORMAL/NORMAL2-IM-0540-0001.jpeg')

imagem_pneumonia = carregar_imagem('chest_xray_dataset/train/PNEUMONIA/person878_bacteria_2801.jpeg')

#alterar o formato da imagem de teste


#ver os valores de cada pixel de image_teste
#normalizando esses valores na escala de 0 - 1
imagem_normal = formatar_imagem(imagem_normal)
imagem_pneumonia = formatar_imagem(imagem_pneumonia)

#alterando o formato para o tensor flow adicionando mais uma coluna

#realizado essas configurações já podemos realizar a previsão
previsao_normal = model.predict(imagem_normal)
previsao_pneumonia = model.predict(imagem_pneumonia)
print(previsao_normal)
print(previsao_pneumonia)
