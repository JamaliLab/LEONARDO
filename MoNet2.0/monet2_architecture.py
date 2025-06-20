
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, Input, GlobalMaxPooling1D, concatenate




def create_model(input_shape=(199, 2), f=32, initializer='he_normal'):
    # Define the input layer
    inputs = Input(shape=input_shape)

    # First branch (x1)
    x1 = Conv1D(f, 4, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f, 4, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f, 4, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalMaxPooling1D()(x1)

    # Second branch (x2)
    x2 = Conv1D(f, 2, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f, 2, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f, 2, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalMaxPooling1D()(x2)

    # Third branch (x3)
    x3 = Conv1D(f, 3, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f, 3, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f, 3, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalMaxPooling1D()(x3)

    # Fourth branch (x4)
    x4 = Conv1D(f, 10, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f, 10, dilation_rate=4, padding='causal', activation='relu', kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f, 10, dilation_rate=8, padding='causal', activation='relu', kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalMaxPooling1D()(x4)

    # Fifth branch (x5)
    x5 = Conv1D(f, 20, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = Conv1D(f, 20, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x5)
    x5 = BatchNormalization()(x5)
    x5 = Conv1D(f, 20, dilation_rate=8, padding='causal', activation='relu', kernel_initializer=initializer)(x5)
    x5 = BatchNormalization()(x5)
    x5 = GlobalMaxPooling1D()(x5)

    # Sixth branch (x6)
    x6 = Conv1D(f, 20, padding='same', activation='relu', kernel_initializer=initializer)(inputs)
    x6 = BatchNormalization()(x6)
    x6 = GlobalMaxPooling1D()(x6)

    # Seventh branch (x7)
    x7 = Conv1D(f, 20, padding='causal', activation='relu', kernel_initializer=initializer)(inputs)
    x7 = BatchNormalization()(x7)
    x7 = Conv1D(f, 20, dilation_rate=2, padding='causal', activation='relu', kernel_initializer=initializer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = Conv1D(f, 20, dilation_rate=16, padding='causal', activation='relu', kernel_initializer=initializer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = GlobalMaxPooling1D()(x7)

    # Concatenate the branches
    con = concatenate([x1, x2, x3, x4, x5, x6, x7])

    # Fully connected layers
    dense = Dense(512, activation='relu')(con)
    features = Dense(128, activation='relu')(dense)
    
    outputs = Dense(7, activation='softmax')(features)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


