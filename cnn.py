from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from datetime import datetime

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation='relu'))

classifier.add(Dropout(0.1))

classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(0.1))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', 
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1/255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         height_shift_range= 0.07,
                                         zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treinamento = gerador_treinamento.flow_from_directory('chest_xray_dataset/train',
                                                           target_size= (64,64),
                                                           batch_size=32, 
                                                           class_mode ='binary')

base_teste = gerador_teste.flow_from_directory('chest_xray_dataset/test',
                                               target_size= (64,64),
                                               batch_size=32,
                                               class_mode='binary')

classifier.fit_generator(base_treinamento, steps_per_epoch=4000/32,
                            epochs=5, validation_data = base_teste, 
                            validation_steps = 1000/32)

classifier.save('./model')

classifier.save_weights(f"./weights_checkpoints/{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.h5")
