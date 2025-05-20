import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


data = pd.read_csv('bus_data.csv')
labels = data['Pathology']
le = LabelEncoder()
y = le.fit_transform(labels)
print(len(y))

seg_dir = "segmented_images"

seg_images = [
    os.path.join(seg_dir, fname)
    for fname in os.listdir(seg_dir)
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
]


def load_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=target_size)
        img_array = preprocess_input(img_to_array(img)) 
        images.append(img_array) 
    return np.array(images)

image_arrays = load_images(seg_images)

X_train, X_val, y_train, y_val = train_test_split(
    image_arrays,
    y,
    test_size=0.3,
    stratify=y,  
    random_state=42
)


original_class_dist = np.unique(y, return_counts=True)
print("Original distribution:", original_class_dist)

val_class_dist = np.unique(y_val, return_counts=True)
print("Validation distribution:", val_class_dist)

original_ratio = original_class_dist[1][1]/original_class_dist[1][0]
val_ratio = val_class_dist[1][1]/val_class_dist[1][0]
print(f"Original malignant/benign ratio: {original_ratio:.2f}")
print(f"Validation malignant/benign ratio: {val_ratio:.2f}")

print("\nValidation set class distribution:")
print(pd.Series(y_val).value_counts(normalize=True))


classes = np.unique(y)
print("\nOriginal class weights:")
pre_aug_weights = compute_class_weight('balanced', classes=classes, y=y)
print(dict(enumerate(pre_aug_weights)))


malignant_images = X_train[y_train == 1]  
benign_images = X_train[y_train == 0]

malignant_images = np.array(malignant_images)
benign_images = np.array(benign_images)

print("Malignant images shape:", malignant_images.shape)  
print("Data type:", malignant_images.dtype)  

aug = ImageDataGenerator(rotation_range= 15, width_shift_range= 0.15, zoom_range= 0.15, fill_mode= 'reflect')

augmented_malignant = []
num_augmented = int(2.5 * len(malignant_images))
aug_gen = aug.flow(malignant_images, batch_size=32, shuffle=False)

for  _ in range(num_augmented // 32 + 1):
    batch = next(aug_gen)
    augmented_malignant.extend(batch)
    
augmented_malignant = np.array(augmented_malignant[:num_augmented]) 
combined_malignant = np.concatenate([malignant_images, augmented_malignant])
balanced_images   = np.concatenate([combined_malignant, benign_images])
balanced_labels   = np.concatenate([
    np.ones(len(combined_malignant)),
    np.zeros(len(benign_images))
])

print("Final class distribution:")
print("Malignant:", len(combined_malignant))
print("Benign:", len(benign_images))

print("Class distribution:", np.unique(balanced_labels, return_counts=True))


print("Total samples:", len(balanced_images))  
print("Val percentage:", len(y_val)/len(balanced_images))

X_train = balanced_images 
y_train = balanced_labels

classes = np.unique(balanced_labels)
class_weights = compute_class_weight('balanced', classes=classes, y=balanced_labels)
class_weights = {i:w for i,w in enumerate(class_weights)}
print("Updated class weights:", class_weights) 

train_datagen = ImageDataGenerator(
    rotation_range=15,  
    width_shift_range=0.10,  
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode='reflect',
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=32,
    shuffle=True
)


X_val = preprocess_input(X_val) 

print("Train min/max:", X_train.min(), X_train.max())  
print("Val min/max:", X_val.min(), X_val.max())   

def build_resnet50():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = True  
    for layer in base.layers[:100]:
        layer.trainable = False
    
    inputs = Input(shape=(224, 224, 3))
    x = base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x) 
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model


model = build_resnet50()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

def focal_loss(gamma=2., alpha=0.5):
    def loss_fn(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt + 1e-7))
    return loss_fn


model.compile(
    optimizer=Adam(3e-4),  
    # loss=focal_loss(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp')
    ]
)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
    ModelCheckpoint('7_best_model.keras', save_best_only=True, monitor='val_auc'),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max')
]

# callbacks = [
#     EarlyStopping(patience=6, restore_best_weights=True),
#     ModelCheckpoint('7_best_model.h5', save_best_only=True)
# ]

steps_per_epoch = math.ceil(len(X_train) / 32)


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val, y_val),
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks
)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.legend()
plt.show()


y_pred = model.predict(X_val) > 0.3
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

fpr, tpr, _ = roc_curve(y_val, model.predict(X_val))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


def plot_predictions(images, true_labels, model, n=5):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        img = (images[i] + [123.68, 116.779, 103.939]) / 255.
        pred = model.predict(img[np.newaxis, ...])[0][0]
        plt.imshow(img)
        plt.title(f"True: {true_labels[i]}\nPred: {pred:.2f}")
        plt.axis('off')
    plt.show()


bc_model = keras.models.load_model('7_best_model.keras', compile=False)

plot_predictions(X_val[:5], y_val[:5], bc_model)
