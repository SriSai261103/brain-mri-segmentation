from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Multiply, Activation
from tensorflow.keras.models import Model

# Nested U-Net (U-Net++)
def nested_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Downsampling path
    c1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    # More U-Net layers would go here, following the skip connections concept

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c1)  # Modify with skip connections as needed

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Attention Gate for Attention U-Net
def attention_block(x, g):
    f = Conv2D(32, (1, 1))(x)
    g1 = Conv2D(32, (1, 1))(g)
    psi = Activation('sigmoid')(f + g1)
    x_out = Multiply()([x, psi])
    return x_out

# Attention U-Net
def attention_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Downsampling path
    c1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    # Attention mechanism
    attention = attention_block(c1, p1)

    # Upsampling path would follow with more layers and skip connections

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(attention)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
