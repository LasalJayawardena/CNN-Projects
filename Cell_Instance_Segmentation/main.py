from google.colab import drive
drive.mount('/content/drive')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/train images',
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical')

print(train_generator.image_shape)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/test images',
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical')

print(test_generator.image_shape)

#plot images :)

from matplotlib import pyplot

x=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])

# plot, first of few images
for i in range(9):

    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

import keras
from keras import layers
from keras import utils
from keras import Sequential

function = Sequential()

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = function(inputs)


    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x 

    for size in [128, 256, 552, 945]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual]) 
        previous_block_activation = x 

    x = layers.SeparableConv2D(1540, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

from keras.utils.vis_utils import plot_model

model = make_model(input_shape=(32,32,3), num_classes=9)
plot_model(model, show_shapes=True)

model.compile(
       optimizer='adam',
       loss="categorical_crossentropy",
       metrics=['accuracy'])

print(train_generator)

model.fit(train_generator,batch_size=16,epochs=200,shuffle=True)

model.save('final_model.h5')


model.summary()

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict(test_generator,nb_samples)

print(predict)

import cv2
from keras.models import load_model

import numpy as np

img = cv2.imread('/content/drive/MyDrive/BV2.jpg')
img = cv2.resize(img,(32,32))
img = np.reshape(img,[1,32,32,3])

modell = load_model('/content/drive/MyDrive/final_model.h5')


classes = modell.predict(img)


for class_name in classes[0]:
  if 1.0 == classes[0][0]:
    print("Image in A172 Class")
  elif 1.0 == classes[0][1]:
    print("Image in BT474 Class")
  elif 1.0 == classes[0][2]:
    print("Image in BV2 Class")
    break
  elif 1.0 == classes[0][3]:
    print("Image in Huh7 Class")
  elif 1.0 == classes[0][4]:
    print("Image in MCF7 Class")
  elif 1.0 == classes[0][5]:
    print("Image in RatC6 Class")
  elif 1.0 == classes[0][6]:
    print("Image in SHSY5Y Class")
  elif 1.0 == classes[0][7]:
    print("Image in SkBr3 Class")
  else:
    print("Image in SKOV3 Class")

!pip install pixellib

import pixellib
from pixellib.semantic import semantic_segmentation 
segment_image = semantic_segmentation()

imgg = cv2.imread('/content/drive/MyDrive/test images/BT474/BT474_Phase_D3_1_00d00h00m_1.tif')
imgg = cv2.resize(imgg,(704,520))
print(imgg.shape)

from PIL import Image
import numpy as np

image = Image.fromarray(imgg)
image.save('testtt.png')
image.show()

from keras.models import load_model

models = load_model('/content/drive/MyDrive/final_model.h5')

modelss = semantic_segmentation(models)

modelss.segmentAsPascalvoc("/content/testtt.png", output_image_name = "image_new.png")
