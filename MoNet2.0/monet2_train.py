import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,CSVLogger
import random
from monet2_architecture import create_model
from tensorflow.keras.utils import to_categorical


device = 'GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU'
print(device)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Call the function to set the random seed
set_random_seed(0)  

steps_per_epoch = 1000


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')

# print(x_train.shape)
# print(y_train.shape)

# # Ensure labels are one-hot encoded
# y_train = to_categorical(y_train, num_classes=3)
# y_val = to_categorical(y_val, num_classes=3)


model = create_model()
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_lr=1e-9),
    ModelCheckpoint(filepath='monet2_1.keras', monitor='val_loss', save_best_only=True, mode='min', save_weights_only=False)
]

history = model.fit(x_train, y_train, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1, callbacks=callbacks, validation_data=(x_val,y_val))