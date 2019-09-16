##
# Chargement de la librairie de tensorflow et de ses dépendances
from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import pathlib
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from sklearn.model_selection import train_test_split
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
##
# Gestion du Dataset
Validation_data_dir = os.getcwd()+"/datasets/valid_set/"
Train_Data_dir = os.getcwd()+"/datasets/train_set/"
Test_data_dir = os.getcwd()+"/datasets/images_test/"
_DATA_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
train_dataset_url = _DATA_URL
# Download et unzip le dataset dans le repertoire root de kears
train_dataset_fp = keras.utils.get_file('101_ObjectCategories', train_dataset_url, untar=True)
# print("Local copy of the dataset file: {}".format(train_dataset_fp))
DATASET = pathlib.Path(train_dataset_fp)
all_images = list(DATASET.glob('*/*'))
all_images = [str(path) for path in all_images]
# Récupération de la liste des nom à attribuer au classes d'images
label_names = sorted(item.name for item in DATASET.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_images]
all_image_labels = np.array(all_image_labels)
# Split les images en Train et Validation
x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
                    train_test_split(all_images, all_image_labels, test_size=0.2, random_state=42)
# Si le dossier Dataset n'existe pas le créer et envoyer les fichiers splités
if not pathlib.Path.is_dir(pathlib.Path(Validation_data_dir))and \
        not pathlib.Path.is_dir(pathlib.Path(Train_Data_dir)):
    for item in all_images:
        tf.gfile.MakeDirs(Validation_data_dir + pathlib.Path(item).parent.name)
        tf.gfile.MakeDirs(Train_Data_dir + pathlib.Path(item).parent.name)
    print("creating validation sets...")
    for val_dir in x_val_filenames:
        tf.gfile.Copy(val_dir, Validation_data_dir + pathlib.Path(val_dir).parent.name + "/" + pathlib.Path(val_dir).name)
    print("creating train sets...")
    for train_dir in x_train_filenames:
        tf.gfile.Copy(train_dir, Train_Data_dir + pathlib.Path(train_dir).parent.name+"/"+pathlib.Path(train_dir).name)
##
if keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, 150, 150)
else:
    input_shape = (150, 150, 3)


def my_model():
    classifier = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=input_shape, activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(102, activation=tf.nn.softmax)
    ])
    classifier.compile(optimizer=keras.optimizers.Adamax(),
                       loss=keras.losses.categorical_crossentropy,
                       metrics=['accuracy'])
    classifier.summary()
    return classifier


Classifier = my_model()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
train_image_data = train_datagen.flow_from_directory(
    Train_Data_dir, batch_size=64, target_size=(150, 150))
valid_image_data = valid_datagen.flow_from_directory(
    Validation_data_dir, batch_size=64, target_size=(150, 150))
STEP_SIZE_TRAIN = train_image_data.n // train_image_data.batch_size
STEP_SIZE_VALID = valid_image_data.n // valid_image_data.batch_size

with tf.device('/device:GPU:0'):
    if not tf.gfile.Exists('Classifier.h5'):
        Classifier.fit_generator(
            train_image_data,
            steps_per_epoch=STEP_SIZE_TRAIN,
            epochs=50,
            validation_data=valid_image_data,
            validation_steps=STEP_SIZE_VALID)
        Classifier.save_weights("Classifier.h5")
    else:
        Classifier.load_weights("Classifier.h5")
        test_images = list(pathlib.Path(Test_data_dir).glob('*/'))
        test_images = [str(path) for path in test_images]
        label_names = sorted(valid_image_data.class_indices.items(), key=lambda pair: pair[1])
        label_names = np.array([key.title() for key, value in label_names])
        test_image_data = []


        def plot_image(idn, predictions_array, img):
            predictions_array, img = predictions_array[idn], img[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(load_img(img))
            predicted_label = np.argmax(predictions_array)
            color = 'blue'
            if np.max(predictions_array) < 0.5:
                color = 'red'

            plt.xlabel("({}) - {:2.0f}%".format(label_names[predicted_label],
                                                100 * np.max(predictions_array)),
                       color=color)


        def plot_value_array(i, predictions_array):
            predictions_array = predictions_array[i]
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            thisplot = plt.bar(range(train_image_data.num_classes), predictions_array, color="blue")
            plt.ylim([0, 1])
            predicted_label = np.argmax(predictions_array)
            thisplot[predicted_label].set_color('green')


        for test_image in test_images:
            test_image_img = load_img(test_image, target_size=(150, 150))
            test_image_array = img_to_array(test_image_img)/255.0
            test_image_data.append(test_image_array)
        test_image_data = np.array(test_image_data)
        result = Classifier.predict(test_image_data)
        num_rows = len(test_images)
        num_cols = 2
        plt.figure()
        for i in range(len(test_images)):
            print("1) {} {:2.0f}%) 2) {} {:2.0f}%\n".format(label_names[np.argsort(result[i])[-1]],
                                                            100 * result[i][np.argsort(result[i])[-1]],
                                                            label_names[np.argsort(result[i])[-2]],
                                                            100 * result[i][np.argsort(result[i])[-2]]))
            plt.subplot(num_rows/2+1, num_cols, i + 1)
            plot_image(i, result, test_images)
        plt.show()
##

