"""Este código carga y procesa datos de imágenes y atributos 
desde el conjunto de datos CelebA, define y entrena un modelo de red neuronal convolucional 
utilizando TensorFlow y Keras."""

#Importación de Librerias
import tensorflow as tf
from keras import Model
from keras import layers, models
from keras.optimizers.legacy import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

#Debo cargar la nueva base de datos con imágenes etiquetadas con imágenes de mi rostro (1) 
# e imagenes de rostros que no sean el mío (0)

#Para evitar problemas con OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.set_printoptions(precision=4)

path_to_cara_images = '/Users/jxel/Reconocimiento-Facial/Cara/'


df_cara = pd.read_csv("caraono.csv", sep=',', header=None)


#Crea conjuntos de datos para nombres de archivos y atributos 
files_cara = tf.data.Dataset.from_tensor_slices(df_cara[0])

attributes_cara = tf.data.Dataset.from_tensor_slices(df_cara.iloc[:,1:].to_numpy())


#Combina los conjuntos de datos en uno solo
data_cara = tf.data.Dataset.zip((files_cara, attributes_cara))


#Se procesan las imágenes con sus atributos
def process_file_cara(file_name, attributes):
    image = tf.io.read_file(path_to_cara_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0 
    return image, attributes

#Aplica la función de procesamiento a cada elemento del conjunto de datos
labeled_images = data_cara.map(process_file_cara)



# Dividir los datos en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(df_cara, test_size=0.2)


# Crear conjuntos de datos para entrenamiento y prueba
train_files = tf.data.Dataset.from_tensor_slices(train_df[0])
train_attributes = tf.data.Dataset.from_tensor_slices(train_df.iloc[:, 1:].to_numpy())
train_data = tf.data.Dataset.zip((train_files, train_attributes))

test_files = tf.data.Dataset.from_tensor_slices(test_df[0])
test_attributes = tf.data.Dataset.from_tensor_slices(test_df.iloc[:, 1:].to_numpy())
test_data = tf.data.Dataset.zip((test_files, test_attributes))

# Aplica la función de procesamiento a cada elemento del conjunto de datos de entrenamiento
train_labeled_images = train_data.map(process_file_cara)


#Visualiza las dos primeras imágenes del conjunto de datos.
for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()



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

"""
# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo en los datos de CelebA
labeled_images = labeled_images.shuffle(buffer_size=10000)  # Aleatorizar el conjunto de datos
labeled_images = labeled_images.batch(32)  # Definir el tamaño del lote
model.fit(labeled_images, epochs=20)

"""
# Extraer las capas convolucionales del modelo entrenado
conv_layers = model.layers[:-2]  # Excluir las dos capas densas (clasificador)

# Crear un nuevo modelo solo con las capas convolucionales
feature_extractor = models.Sequential(conv_layers)

# Congelar las capas convolucionales (parte preentrenada)
for layer in feature_extractor.layers:
    layer.trainable = False

# Crear un nuevo modelo para clasificación binaria (rostro o no rostro)
new_classifier = models.Sequential([
    # Agregar capas densas para la clasificación binaria
    layers.Flatten(input_shape=(feature_extractor.output_shape[1:])),  # Aplanar las salidas convolucionales
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Capa de salida con una neurona y función de activación sigmoide para clasificación binaria
])

# Combinar el extractor de características y el nuevo clasificador
new_model = models.Sequential([
    feature_extractor,
    new_classifier
])

# Compilar el nuevo modelo
new_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Visualizar la arquitectura del nuevo modelo
new_model.summary()

#Entrenar el modelo con los nuevos datos de mi cara
train_labeled_images = train_labeled_images.shuffle(buffer_size=10000) #Aleatorizar el conjunto de entreanmiento
train_labeled_images = train_labeled_images.batch(32) #Tamaño del lote

#Entrenamos el nuevo modelo con el nuevo clasificador
new_model.fit(train_labeled_images, epochs=40)