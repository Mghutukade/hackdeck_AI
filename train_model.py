import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import image_dataset_from_directory


IMG_SIZE = (224, 224)
BATCH_SIZE = 32  #batch of 32 imgs
TRAIN_TIME= 5  #how many times the model will work....
DATA_PATH = "dataset"


train_ds = image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2, #*0% data for training and 20% for validation
    subset="training", #calls only training 
    seed=123, # fixed data splits 
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)


# CNN MODEL (MobileNetV2)- better for mobile

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False, # remove old data - defualt data
    weights="imagenet"
)

base_model.trainable = False  # freezing CNN layers - convert img to feature map

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TRAIN_TIME
)


model.save("crop_disease_model.keras")

print(" Model trained and saved successfully!")


# Crop Disease Detection - Model Training Script

# This script loads crop leaf images from folders,
# uses transfer learning with MobileNetV2,
# trains a CNN classifier, and saves the model
# for later use in a web application.
