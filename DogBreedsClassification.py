
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
import random
import cv2
import time
from collections import Counter
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Sequential
from keras import metrics
from keras.utils import multi_gpu_model
import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


# # 1. Load Data
# ## 1.1 Train labels

# In[2]:
df = pd.read_csv("labels.csv")
df_valid = df.loc[int(len(df.index) * 0.8):]
df = df.loc[:int(len(df.index) * 0.8)]

nTrain = len(df.index)
nValid = len(df_valid.index)

print("Number of training samples:", nTrain)
print("Number of validation samples:", nValid)
df.head()


# In[3]:


breeds = list(set(df.breed))
breed_counts = dict(Counter(df.breed.values))


# In[4]:


_= plt.figure(figsize=(20, 10))
g = sns.countplot(x="breed", data=df, orient='v')
g.set_xticklabels(g.get_xticklabels(), rotation="vertical");
g.tick_params(labelsize=8)
plt.show()


# In[5]:


bins = list(range(60, 140, 10))

groups = df.groupby("breed").count().groupby(["breed", pd.cut(df.groupby("breed").count().id, bins)])
groups.size().unstack()
data = groups.size().unstack().sum().to_frame().reset_index().rename(columns={0: "Frequency", "id": "Counts"})

df_stats = df.groupby("breed").count().reset_index().rename(columns={"id": "counts"})
print("Less freq breed: {} counts: {}".format(df_stats.min().breed, df_stats.min().counts))
print("Max freq breed: {} counts: {}".format(df_stats.max().breed, df_stats.max().counts))
print("Mean: {}, Std: {:.1f}\n".format(int(df_stats.counts.mean()), float(df_stats.counts.std())))

_ = plt.figure(figsize=((8, 6)));
sns.barplot(x="Counts", y="Frequency", data=data)
plt.show()


# ## 1.2 Train examples

# In[4]:


def read_image(img_path, size=(224, 224)):
    img = image.load_img(img_path, target_size=size)
    return image.img_to_array(img)


# In[7]:


fig = plt.figure(1, figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=.05)

for i, row in df.sample(9, axis=0).reset_index().iterrows():
    ax = grid[i]
    img = read_image("train/{}/{}.jpg".format(row.breed, row.id))
    ax.imshow(img / 255.)
    ax.text(20, 200, "Label: {}".format(row.breed), backgroundcolor="w", color="k", alpha=0.8)
    ax.axis("off")


# In[8]:


vgg = VGG16(weights="imagenet")

fig = plt.figure(1, figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=.05)

for i, row in df.sample(9, axis=0).reset_index().iterrows():
    ax = grid[i]
    img = read_image("train/{}/{}.jpg".format(row.breed, row.id))
    ax.imshow(img / 255.)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    preds = vgg.predict(x)
    _, class_pred, score = decode_predictions(preds, 1)[0][0]
    ax.text(20, 180, "VGG16: {} {:.2f}".format(class_pred, float(score)), color='w', backgroundcolor='black', alpha=0.8)
    ax.text(20, 200, "Label: {}".format(row.breed), color='black', backgroundcolor="w", alpha=0.8)
    ax.axis("off")


# # 2. Define Model

# ## 2.1 Data Generator

# In[5]:
def preprocessing_function(img):
        return preprocess_input(img)


batch_size = 64

train_datagen = ImageDataGenerator(
    rotation_range=40, 
    width_shift_range=.2, 
    height_shift_range=.2,
    rescale = 1. / 255,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=preprocessing_function)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory("train/", target_size=(224, 224), batch_size=batch_size)
valid_generator = valid_datagen.flow_from_directory("valid/", target_size=(224, 224), batch_size=batch_size)

#%%

print(next(train_generator))

# ## 2.2 Load pre-trained Features

# In[6]:


vgg_convs = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='max')
valid_features = vgg_convs.predict_generator(train_generator, 
                                             verbose=1)


# In[20]:


def pre_train_processing(generator):
    vgg_convs = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling='max')
    # vgg_convs = multi_gpu_model(vgg_convs, 8)
    
    features = np.zeros(shape=(generator.n, 7, 7, 512))
    labels = np.zeros(shape=(generator.n, 120))
    i = 0
    for inputs_batch, labels_batch in tqdm(generator):
        features_batch = vgg_convs.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= generator.n:
            break
    return np.reshape(features, (generator.n, 7 * 7 * 512)), labels

train_features, train_labels = pre_train_processing(train_generator)
valid_features, valid_labels = pre_train_processing(valid_generator)


# ## 2.3 Classifier Model

# In[15]:


del model
model = Sequential()
model.add(Dense(256, activation="relu", input_dim=7 * 7 * 512))
model.add(Dropout(0.5))
model.add(Dense(120, activation="softmax"))
# model = multi_gpu_model(model, 8)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


now = time.strftime("%Y%m%d_%H%M%S")
step_per_epochs = train_generator.n // batch_size
tensorboard = TensorBoard(log_dir="logs/" + now, write_graph=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=10)
history = model.fit(train_features,
                    train_labels, epochs=500, 
                    batch_size=batch_size,
                    # steps_per_epoch=train_generator.n // batch_size,
                    validation_data=(valid_features, valid_labels),
                    # validation_steps=valid_generator.n // batch_size,
                    callbacks=[tensorboard])


# # 4. Predictions

# In[ ]:


def encode_label(breed, generator):
    y_true = np.zeros((1, 120))
    y_true[0][generator.class_indices[breed]] = 1.
    return y_true

def decode_label(y_pred, generator):
    reversed_class_indeces = dict(map(reversed, generator.class_indices.items()))
    return reversed_class_indeces[np.argmax(y_pred)]

def decode_prediction(pred, generator):
    y_true = encode_label(row.breed, generator)
    class_pred = decode_label(pred, generator)
    acc = K.eval(metrics.categorical_accuracy(y_true, y_pred))
    
    return class_pred, acc


# In[ ]:


fig = plt.figure(3, figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=.05)
vgg_convs = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for i, row in df_valid.sample(9, axis=0).reset_index().iterrows():
    ax = grid[i]
    img = read_image("valid/{}/{}.jpg".format(row.breed, row.id))
    ax.imshow(img / 255.)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    x = vgg_convs.predict(x)
        
    y_pred = model.predict(x=x.reshape(1, 7 * 7 * 512))
    class_pred, score = decode_prediction(y_pred, valid_generator)

    ax.text(20, 180, "Prediction: {} {:.2f}".format(class_pred, float(score)), color='w', backgroundcolor='black', alpha=0.8)
    ax.text(20, 200, "Label: {}".format(row.breed), color='black', backgroundcolor="w", alpha=0.8)
    ax.axis("off")


# In[ ]:


i = 455

img = read_image("valid/{}/{}.jpg".format(df_valid.iloc[i].breed, df_valid.iloc[i].id))
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
x = vgg_convs.predict(x)
pred = model.predict(x.reshape(1, 7 * 7 * 512))
pred


# In[ ]:


y_true.reshape(1, 120)


# In[ ]:


pred


# In[ ]:


x.reshape(7 * 7 * 512).shape
valid_generator.c


# In[ ]:


decode_label(118, valid_generator)


