import os
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.preprocessing import image

df = pd.read_csv("labels.csv")
df = df.loc[len(df.index) * 0.8:]

df_sample = df.sample(9, axis=0)


def read_img(img_path, size=(224, 224)):
    img = image.load_img(img_path, target_size=size)
    return image.img_to_array(img)


model = ResNet50(weights='imagenet')

fig = plt.figure(1, figsize=(12, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=.05)
for i, row in df_sample.reset_index().iterrows():
    ax = grid[i]
    img = read_img("valid/{}/{}.jpg".format(row.breed, row.id))
    ax.imshow(img / 255.)
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    preds = model.predict(x)
    _, class_pred, score = decode_predictions(preds, 1)[0][0]
    ax.text(20, 180, "ResNet50: {} {:.2f}".format(class_pred, float(score)), color='w', backgroundcolor='black', alpha=0.8)
    ax.text(20, 200, "Label: {}".format(row.breed), color='black', backgroundcolor="w", alpha=0.8)
    ax.axis("off")
plt.show()