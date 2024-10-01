import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from models import nested_unet, attention_unet
from preprocessing import preprocess_image

# Load your dataset here (example)
def load_data():
    # Assume images and masks are already preprocessed and loaded as numpy arrays
    # images = ...
    # masks = ...
    return train_test_split(images, masks, test_size=0.2, random_state=42)

# Compile, train, and evaluate the model
def compile_and_train(model, train_images, train_masks, val_images, val_masks):
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=[MeanIoU(num_classes=2)])

    model.fit(train_images, train_masks, validation_data=(val_images, val_masks),
              epochs=50, batch_size=16)
    
    return model

# Load data
train_images, val_images, train_masks, val_masks = load_data()

# Select and train the model (choose Nested U-Net or Attention U-Net)
model = nested_unet(input_shape=(256, 256, 1))  # or attention_unet()
model = compile_and_train(model, train_images, train_masks, val_images, val_masks)

# Save the trained model
model.save('best_model.h5')
