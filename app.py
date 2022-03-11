import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


# loading the four best models
DM_novel = keras.models.load_model('cnn_derma_novel_balanced.h5')
DM_AUG = keras.models.load_model('cnn_derma_aug.h5')
RM_novel = keras.models.load_model('cnn_retina_novel_balanced.h5')
RM_AUG = keras.models.load_model('cnn_retina_aug.h5')

# import os
# creating a function to view the image file

# creating the function to predict the image (adapted from source [11])
def img_load(img):
  # Create the array of the right shape to feed into the keras model
  img_data = np.ndarray(shape=(1,28,28,3), dtype=np.float32)
  image = img
  #image sizing
  size = (28, 28)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)

  #turn the image into a numpy array
  img_array = np.asarray(image)
  # Normalize the image
  normalized_img_array = (img_array.astype(np.float32) / 255.0)

  # Load the image into the array
  img_data[0] = normalized_img_array
  return img_data

def main():
  # creating a list off all uploaded images
  st.title('Medical Image Analysis and Classification')
    
  menu = ['DermaMNIST','RetinaMNIST']
  choice = st.sidebar.selectbox('Choose a Dataset',menu)
  
  if choice == 'DermaMNIST':
    st.subheader('DermaMNIST Machine Learning Models')
    upload_image = st.file_uploader('Upload Images', type=['jpg','jpeg'])
    if upload_image is not None:
      # opening the image 
      image_1 = Image.open(upload_image)
      st.image(image_1, caption='Uploaded Image.', use_column_width=False)
      img_1 = img_load(image_1)
      
      st.write('CNN with Balanced Data :')
      DM_nov = DM_novel.predict(img_1)
      acc = np.amax(DM_nov)
      class_predict = np.argmax(DM_nov) # gives the prediction in an integer value
      st.write('  Class predicted: ', class_predict)
      st.write('  Accuracy: ', acc)
      
      st.write('CNN with Augmented Data:')
      DM_aug = DM_AUG.predict(img_1)
      acc = np.amax(DM_aug)
      class_predict = np.argmax(DM_aug) # gives the prediction in an integer value
      st.write('  Class predicted: ', class_predict)
      st.write('  Accuracy: ', acc)
  
  elif choice == 'RetinaMNIST':
    st.subheader('RetinaMNIST Machine Learning Models')
    upload_image= st.file_uploader('Upload Images', type=['jpg','jpeg'])
    if upload_image is not None:
      # opening the image 
      image_2 = Image.open(upload_image)
      st.image(image_2, caption='Uploaded Image.', use_column_width=False)
      img_2 = img_load(image_2)
      
      st.write('CNN with Balanced Data:')
      RM_nov = RM_novel.predict(img_2)
      acc = np.amax(RM_nov)
      class_predict = np.argmax(RM_nov) # gives the prediction in an integer value
      st.write('  Class predicted: ', class_predict)
      st.write('  Accuracy: ', acc)

      st.write('CNN with Augmented Data:')
      RM_aug = RM_AUG.predict(img_2)
      acc = np.amax(RM_aug)
      class_predict = np.argmax(RM_aug) # gives the prediction in an integer value
      st.write('  Class predicted: ', class_predict)
      st.write('  Accuracy: ', acc)
    
  
if __name__ == '__main__':
  main()