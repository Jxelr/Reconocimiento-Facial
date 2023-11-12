"""Este código carga y procesa datos de imágenes y atributos 
desde el conjunto de datos CelebA, define y entrena un modelo de red neuronal convolucional 
utilizando TensorFlow y Keras."""

#Importación de Librerias
import tensorflow as tf
from keras import layers, models
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

#Para evitar problemas con OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.set_printoptions(precision=4)


#Eliminar el doble espacio entre algunos datos de la tabla
with open('list_attr_celeba.txt', 'r') as f:
   print("skipping : " + f.readline())
   print("skipping headers : " + f.readline())
   with open('attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            newf.write(new_line)
            newf.write('\n')

#Lee el archivo ya corregido y lo carga a un dataframe de pandas
df = pd.read_csv("attr_celeba_prepared.txt" , sep=' ', header = None)

"""
print("----------")
print(df[0].head())
print(df.iloc[:,1:].head())
print("----------")
print(df.head())
#exit()
"""


#Crea conjuntos de datos para nombres de archivos y atributos 
files = tf.data.Dataset.from_tensor_slices(df[0])

attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())


#Combina los conjuntos de datos en uno solo
data = tf.data.Dataset.zip((files, attributes))


#Ruta del directorio donde están las imágenes
path_to_images = 'img_align_celeba/'


#Se procesan las imágenes con sus atributos
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0 
    return image, attributes

#Aplica la función de procesamiento a cada elemento del conjunto de datos
labeled_images = data.map(process_file)

# print(labeled_images)


#Visualiza las dos primeras imágenes del conjunto de datos.
for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()

# Dividir los datos en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(df, test_size=0.2)

# Definir la arquitectura de la red convolucional
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192, 192, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(40, activation='sigmoid'))  # Capa de salida con 40 neuronas (40 atributos)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo en los datos de CelebA
labeled_images = labeled_images.shuffle(buffer_size=10000)  # Aleatorizar el conjunto de datos
labeled_images = labeled_images.batch(32)  # Definir el tamaño del lote
model.fit(labeled_images, epochs=20)