import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
from collections import Counter
from tqdm import tqdm as tqdm
import random

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.utils import multi_gpu_model
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# # 1. Load Data
# ## 1.1 Train labels

df = pd.read_csv("labels.csv")
df_valid = df.loc[int(len(df.index) * 0.8):]
df = df.loc[:int(len(df.index) * 0.8)]
print("Number of training samples:", len(df.index))
print("Number of validation samples:", len(df_valid.index))
df.head()

breeds = list(set(df.breed))
breed_counts = dict(Counter(df.breed.values))

bins = list(range(60, 140, 10))

groups = df.groupby("breed").count().groupby(["breed", pd.cut(df.groupby("breed").count().id, bins)])
groups.size().unstack()
data = groups.size().unstack().sum().to_frame().reset_index().rename(columns={0: "Frequency", "id": "Counts"})

df_stats = df.groupby("breed").count().reset_index().rename(columns={"id": "count"})
print("Less freq breed: {} counts: {}".format(df_stats.breed.min(), df_stats.min().count()))
print("Max freq breed: {} counts: {}".format(df_stats.breed.max(), df_stats.max().count()))
print("Mean: {}, Std: {:.1f}\n".format(int(df_stats["count"].mean()), float(df_stats["count"].std())))


# ## 1.3 Preprocess Images


def process_img(image, w, h):
    return cv2.resize(image, (w, h))


# ## 1.4 One-hot label vectors

y_true = []
for _, row in df.iterrows():
    lbl = np.zeros(len(breeds))
    lbl[breeds.index(row.breed)] = 1
    y_true.append(lbl)
y_true = np.array(y_true)


# # 2. Define Model
# ## 2.1 Create Model

def create_model(input_shape):
    features = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3),
               activation="relu",
               padding="same",
               name="block1_conv1")(features)
    x = Conv2D(64, (3, 3),
               activation="relu",
               padding="same",
               name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_maxpool")(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation="relu",
               padding="same",
               name="block2_conv1")(x)
    x = Conv2D(128, (3, 3),
               activation="relu",
               padding="same",
               name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_maxpool")(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation="relu",
               padding="same",
               name="block3_conv1")(x)
    x = Conv2D(256, (3, 3),
               activation="relu",
               padding="same",
               name="block3_conv2")(x)
    x = Conv2D(256, (3, 3),
               activation="relu",
               padding="same",
               name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_maxpool")(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation="relu",
               padding="same",
               name="block4_conv1")(x)
    x = Conv2D(512, (3, 3),
               activation="relu",
               padding="same",
               name="block4_conv2")(x)
    x = Conv2D(512, (3, 3),
               activation="relu",
               padding="same",
               name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_maxpool")(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation="relu",
               padding="same",
               name="block5_conv1")(x)
    x = Conv2D(512, (3, 3),
               activation="relu",
               padding="same",
               name="block5_conv2")(x)
    x = Conv2D(512, (3, 3),
               activation="relu",
               padding="same",
               name="block5_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_maxpool")(x)

    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation="relu", name="fc1")(x)
    x = Dense(4096, activation="relu", name="fc2")(x)
    y = Dense(len(breeds), activation="softmax", name="culo")(x)

    return Model(inputs=features, outputs=y, name="vgg16")


def create_model_2(input_shape):
    features = Input(shape=input_shape)
    with K.name_scope('First_convolution'):
        x = Conv2D(filters=10, kernel_size=(5, 5), activation='relu', padding='same')(features)
        x = MaxPooling2D(padding='same')(x)
    with K.name_scope('Second_convolution'):
        x = Conv2D(filters=20, kernel_size=(5, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D(padding='same')(x)
    with K.name_scope('Dropout_Flatten'):
        x = Dropout(0.5)(x)
        x = Flatten()(x)
    with K.name_scope('Fully_connected'):
        x = Dense(120, activation='relu')(x)
        y = Dense(120, activation='softmax')(x)
    return Model(inputs=features, outputs=y)


# ## 2.2 Define loss and accuracy

def loss(y_true, y_pred):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    return tf.reduce_mean(cross_entropy)


def accuracy(y_true, y_pred):
    correct_pred = tf.equal(y_true, y_pred)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# ## 2.3 Create Model

# del model
model = create_model((256, 256, 3))

model = multi_gpu_model(model, 8)


# ## 2.4 Data Generator

def load_data(df, y_true):
    img_paths = ["train/{}.jpg".format(x) for x in df.id]
    imgs = np.array([process_img(cv2.imread(x), 150, 150) for x in tqdm(img_paths)])

    return imgs, y_true, img_paths


# print("Loading...")
# train_images, train_labels, train_paths = load_data(df, y_true)

batch_size = 512
# Compute mean and std on a smaller subsamples to speed up
# datagen = ImageDataGenerator(featurewise_center=True,
#                              featurewise_std_normalization=True,
#                              validation_split=0.2)


train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory("train/", target_size=(256, 256),
                                                    class_mode="categorical")
valid_generator = train_datagen.flow_from_directory("valid/", target_size=(256, 256),
                                                    class_mode="categorical")

# datagen.fit(train_images[:1000])


# train_generator = datagen.flow(x=train_images,
#                                y=train_labels,
#                                batch_size=batch_size,
#                                subset="training")
# valid_generator = datagen.flow(x=train_images,
#                                y=train_labels,
#                                batch_size=batch_size,
#                                subset="validation")


model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
model.summary()

now = time.strftime("%Y%m%d_%H%M%S")
tensorboard = TensorBoard(log_dir="logs/" + now, write_graph=True)
history = model.fit_generator(train_generator,
                              validation_data=valid_generator,
                              steps_per_epoch=len(df.index) // batch_size,
                              validation_steps=len(df_valid.index) // batch_size,
                              epochs=50, verbose=1,
                              callbacks=[tensorboard])

model.save_weights("model.h5", overwrite=True)
print("Done.")

# # # 4. Predictions
#
# rand_samples = random.sample(range(len(df.index)), 9)
#
# samples_paths = df.loc[rand_samples].id.apply(lambda x: "train/{}.jpg".format(x))
# samples_y_true = df.loc[rand_samples].breed.values
# samples_imgs = np.array([process_img(cv2.imread(x), 150, 150) for x in samples_paths])
# predictions = model.predict_generator(datagen.flow(samples_imgs))
#
# _, axs = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
# for i, path in enumerate(samples_paths):
#     img = cv2.imread(path)
#     y_pr = breeds[np.argmax(np.round(predictions[i], 3))]
#     axs[i // 3, i % 3].imshow(img)
#     axs[i // 3, i % 3].set_title("True: {}\nprediction {}".format(samples_y_true[i], y_pr))
#     axs[i // 3, i % 3].axis("off")
