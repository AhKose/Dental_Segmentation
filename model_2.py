import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation, Add, Multiply

# Function to add Convolutional Block with Residual Connection
def conv_block(input_tensor, num_filters):
    # First layer
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Residual connection
    shortcut = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])

    x = Activation('relu')(x)
    return x

# Function for Attention Block
def attention_block(x, shortcut, num_filters):
    g1 = Conv2D(num_filters, (1, 1), padding='same')(x)
    g1 = BatchNormalization()(g1)

    x1 = Conv2D(num_filters, (1, 1), padding='same')(shortcut)
    x1 = BatchNormalization()(x1)

    add = Add()([g1, x1])
    relu = Activation('relu')(add)
    psi = Conv2D(1, (1, 1), padding='same')(relu)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    return Multiply()([x, psi])

# 2nd U-Net Model with increased depth, width, residual connections, and attention mechanisms
def unet(input_size=(256, 256, 1), num_filters=64, dropout=0.5):
    inputs = Input(input_size)

    # Contracting Path
    c1 = conv_block(inputs, num_filters)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv_block(p1, num_filters * 2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv_block(p2, num_filters * 4)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv_block(p3, num_filters * 8)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = conv_block(p4, num_filters * 16)

    # Expansive Path with attention
    u6 = UpSampling2D((2, 2))(c5)
    u6 = attention_block(u6, c4, num_filters * 8)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv_block(u6, num_filters * 8)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = attention_block(u7, c3, num_filters * 4)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv_block(u7, num_filters * 4)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = attention_block(u8, c2, num_filters * 2)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv_block(u8, num_filters * 2)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = attention_block(u9, c1, num_filters)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv_block(u9, num_filters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Instantiate the model
model = unet()
