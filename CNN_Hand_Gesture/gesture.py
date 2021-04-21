import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau


st.markdown("<h1 style='text-align: center;'>Hand Gesture Recognition </h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>L Jay</h3>",unsafe_allow_html=True)

st.text('2. Results of your selection:')

gesture = pd.read_csv('./dataset/sign_mnist_test.csv')

st.sidebar.title('1. Choose a gesture:')

a=st.sidebar.number_input(label='Enter a value upto 255:',min_value=0,value=0,step=1)

    
# Take the label
label = gesture['label'][a]
    
# Take the pixels
pixels = gesture.iloc[a, 1:]

# The pixel intensity values are integers from 0 to 255
pixels = np.array(pixels, dtype='uint8')

# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))
    
    

# Plot    
st.sidebar.text('Label is {label}'.format(label=label))
    
st.sidebar.image(pixels)


#extract labels
y_test=gesture['label']
del gesture['label']


#Label Binarizer

label_binarizer=LabelBinarizer()

y_test=label_binarizer.fit_transform(y_test)

x_test=gesture.values

#normalization

x_test=x_test/255


#reshape

x_test=x_test.reshape(-1,28,28,1)


st.cache(allow_output_mutation=True)
model = tf.keras.models.load_model('hand_gesture.h5')
predictions=model.predict_classes(x_test)
  

st.success('Done!')

#st.image(predictions[a].astype('uint8'),clamp=True)
st.text('The alphabet is {}'.format(chr(ord('`')+label)))


      

