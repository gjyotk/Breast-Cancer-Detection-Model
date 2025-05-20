import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img

image_dir = "Images"
mask_dir = "Masks"


image_filenames = [
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
]

rgb_images = []
for path in image_filenames:
    img = load_img(path, target_size=(256, 256))  
    arr = img_to_array(img)                      
    rgb_images.append(arr)

rgb_images = np.stack(rgb_images, axis=0)    
print("Loaded images:", rgb_images.shape)


def dice_score(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

seg_model = keras.models.load_model('saved_unet_model.keras', custom_objects={'dice_score': dice_score})
seg_model.trainable = False

seg_images=[]

for image in rgb_images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        image = image.astype('float32') / 255.0          
        input_img = np.expand_dims(image, axis=0)               

        mask_prob = seg_model.predict(input_img)                 
        mask_bin = (mask_prob > 0.5).astype('float32') 
        seg_images.append(image * 0.5 + mask_bin[0] * 0.5 )

        
print(seg_images[0].shape)
print("Min pixel:", seg_images[0].min(), "Max pixel:", seg_images[0].max())


output_dir = 'segmented_images'
os.makedirs(output_dir, exist_ok=True)

for idx, seg in enumerate(seg_images):
    seg_uint8 = (seg * 255).clip(0,255).astype(np.uint8)
    bgr = cv2.cvtColor(seg_uint8, cv2.COLOR_RGB2BGR)
    fname = f'seg_{idx:04d}.png'
    path  = os.path.join(output_dir, fname)
    cv2.imwrite(path, bgr)

print(f"Saved {len(seg_images)} segmented images to `{output_dir}/`")