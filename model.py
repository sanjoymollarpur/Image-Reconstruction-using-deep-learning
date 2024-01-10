import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam





def unet(input_size=(512, 512, 1), learning_rate=0.001):
    inputs = tf.keras.Input(shape=input_size)

    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv4)

    up7 = layers.UpSampling2D(size=(2, 2))(conv4)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(up7)

    merge7 = layers.concatenate([conv7, conv3], axis=-1)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv7)

    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(up8)

    merge8 = layers.concatenate([conv8, conv2], axis=-1)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv8)

    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(up9)

    merge9 = layers.concatenate([conv9, conv1], axis=-1)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(merge9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='glorot_uniform')(conv9)

    output = layers.Conv2D(1, 1, activation=None)(conv9)

    model = models.Model(inputs=inputs, outputs=output)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def main():
    
    model = unet(input_size=(512, 512, 1), learning_rate=0.001)  # Assuming your unet function is defined
    print(model.summary())

    


if __name__ == "__main__":
    main()


