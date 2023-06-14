import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K

def sampling(args):
    z_mean, z_log_var, latent_dim = args
    print(z_mean)
    print(z_log_var)
    print(latent_dim)
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim),
                                        mean=0., stddev=1.)
    return z_mean + keras.backend.exp(z_log_var) * epsilon

def masking(args):
    input, mask = args

    return tf.where(mask, input, 0)

"""
Class of AutoEncoders
----------------------------------------------------
[int int], int, int -> object
----------------------------------------------------
Inputs:
input_dim: 2-array with the dimensions of the image.
latent_dim: int wit the dimension of the latent 
    space.
arch: the architecture of AutoEncoder to be used.
    The default is the arch=1.
in_channels: input channels to be used. That is, if
    you are using a RGB image as input, it has 3 
    input channels. Also, if you use a ndarray 
    with differents variables (wind, temperature, 
    pressure, etc.), you could use each variable as
    a channel.
out_channels: output channels to be computed. In the
    example of a RBG image as input, you can 
    generate a RGB image (3 output channels) or a
    gray-scale image (1 output channel).
----------------------------------------------------
Output:
After initialization, it generates an object with
the AutoEncoder architecture and methods
"""
class AE_conv():
    def __init__(self, input_dim, latent_dim, arch=1, in_channels=1, out_channels=1, VAE=False, mask=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.VAE = VAE
        if mask is None:
            self.mask = tf.ones(tf.concat([self.input_dim, [self.in_channels]], 0), dtype=bool)
        else:
            self.mask = tf.convert_to_tensor(mask, dtype=bool)
        if arch==1:
            # default architecture
            self.def_arch_build()
        elif arch==2:
            # default simplified
            self.simple_arch_build()
        elif arch==3:
            # batch normalized & simetric
            self.batched_simetric_build()
        elif arch==5:
            # default adapted problem 
            self.busi_arch_build()
        elif arch==6:
            # busi arch simplified
            self.simple_busi_arch_build()
        elif arch==7:
            # dense architecture
            self.dense_build()

    # Default architecture of anti-simetric AutoEncoder
    def def_arch_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))


        # Encoder
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(input_img)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(16, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)


        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Latent space:
        x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Decoder
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(encoded)
        x = layers.Dropout(0.3)(x)

        dim_row, dim_col = self.input_dim[0]/18, self.input_dim[1]/18

        x = layers.Dense(int(dim_row * dim_col), activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Reshape((int(dim_row), int(dim_col), 1))(x)

        x = layers.Conv2DTranspose(128, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)
        x = layers.Conv2DTranspose(64, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)
        x = layers.Conv2DTranspose(32, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)
        
        # Output
        x = layers.Conv2DTranspose(3, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        decoded = layers.Conv2D(self.out_channels, 3, activation='selu', padding='same', strides=1)(x)

        # Model
        self.encoder = keras.Model(input_img, encoded)
        self.decoder = keras.Model(encoded, decoded)
        self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())

    # Simplified architecture of anti-simetric AutoEncoder
    def simple_arch_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))


        # Encoder
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(input_img)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(16, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)


        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Latent space:
        x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Decoder
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(encoded)
        x = layers.Dropout(0.3)(x)

        dim_row, dim_col = self.input_dim[0]/18, self.input_dim[1]/18

        x = layers.Dense(int(dim_row * dim_col), activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Reshape((int(dim_row), int(dim_col), 1))(x)

        x = layers.Conv2DTranspose(128, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=6)(x)
        x = layers.Conv2DTranspose(32, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)
        
        # Output
        x = layers.Conv2DTranspose(3, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        decoded = layers.Conv2D(self.out_channels, 3, activation='selu', padding='same', strides=1)(x)

        # Model
        self.encoder = keras.Model(input_img, encoded)
        self.decoder = keras.Model(encoded, decoded)
        self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())

    # Batch normalization and simetric architecture of AutoEncoder
    def batched_simetric_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))


        # Encoder
        x = layers.Conv2D(64, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(512, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)


        x = layers.Flatten()(x)
        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Latent space:
        x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Decoder
        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.3))(encoded)
        x = layers.Dropout(0.3)(x)

        dim_row, dim_col = self.input_dim[0]/16, self.input_dim[1]/16

        x = layers.Dense(int(dim_row * dim_col), activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Reshape((int(dim_row), int(dim_col), 1))(x)

        x = layers.Conv2DTranspose(512, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(256, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(128, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=2)(x)
        x = layers.BatchNormalization()(x)
        
        # Output
        x = layers.Conv2DTranspose(3, 4, activation=layers.LeakyReLU(alpha=0.2), padding='same', strides=1)(x)
        decoded = layers.Conv2D(self.out_channels, 3, activation='tanh', padding='same', strides=1)(x)

        # Model
        self.encoder = keras.Model(input_img, encoded)
        self.decoder = keras.Model(encoded, decoded)
        self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())

    # Special architecture for the busi
    def busi_arch_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))


        # Encoder
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(input_img)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(16, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(x)


        
        x = layers.Dropout(0.3)(x)

        # Latent space:
        if self.VAE:
            x = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)
            z_mean = layers.Dense(self.latent_dim)(x)
            z_log_var = layers.Dense(self.latent_dim)(x)
            encoded = layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var, self.latent_dim])

            latent_inputs = layers.Input(shape=(self.latent_dim,), name='z_sampling')
            x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(latent_inputs)
        else:
            encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)
            x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(encoded)
        
        # Decoder
        x = layers.Dropout(0.3)(x)

        dim_row, dim_col = self.input_dim[0]/4, self.input_dim[1]/4

        x = layers.Dense(int(dim_row * dim_col), activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Reshape((int(dim_row), int(dim_col), 1))(x)

        x = layers.Conv2DTranspose(128, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2DTranspose(64, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)
        x = layers.Conv2DTranspose(32, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)
        
        # Output
        x = layers.Conv2DTranspose(3, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        decoded = layers.Conv2D(self.out_channels, 3, activation='selu', padding='same', strides=1)(x)

        # Model
        if self.VAE:
            self.encoder = keras.Model(input_img, encoded, name='encoder')
            self.decoder = keras.Model(latent_inputs, decoded)
            outputs = self.decoder(self.encoder(input_img))#[2])
            self.autoencoder = keras.Model(input_img, outputs)
        else:
            self.encoder = keras.Model(input_img, encoded)
            self.decoder = keras.Model(encoded, decoded)
            self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())

    
    # Simple special architecture for the busi
    def simple_busi_arch_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1], self.in_channels))


        # Encoder
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(input_img)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)

        #x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        #x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        #x = layers.Conv2D(16, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Latent space:
        x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)


        # Decoder
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(encoded)
        x = layers.Dropout(0.3)(x)

        dim_row, dim_col = self.input_dim[0]/4, self.input_dim[1]/4

        x = layers.Dense(int(dim_row * dim_col), activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Reshape((int(dim_row), int(dim_col), 1))(x)

        #x = layers.Conv2DTranspose(128, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2DTranspose(64, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)
        x = layers.Conv2DTranspose(32, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=2)(x)
        
        # Output
        x = layers.Conv2DTranspose(3, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        decoded = layers.Conv2D(self.out_channels, 3, activation='selu', padding='same', strides=1)(x)

        # Model
        self.encoder = keras.Model(input_img, encoded)
        self.decoder = keras.Model(encoded, decoded)
        self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())

    def dense_build(self):
        # Input
        input_img = keras.Input(shape=(self.input_dim[0], self.input_dim[1]))

        #  Encoder
        x = layers.Flatten()(input_img)
        x = layers.Dense(1024, activation='selu')(x)
        x = layers.Dense(512, activation='selu')(x)
        x = layers.Dense(128, activation='selu')(x)
        x = layers.Dense(32, activation='selu')(x)

        # Latent space
        #x = layers.Dropout(0.3)(x)
        encoded = layers.Dense(self.latent_dim, activation='selu')(x)

        # Decoder
        x = layers.Dense(32, activation='selu')(encoded)
        x = layers.Dense(128, activation='selu')(x)

        # Output
        x = layers.Dense(self.input_dim[0] * self.input_dim[1], activation='sigmoid')(x)
        decoded = layers.Reshape((self.input_dim[0], self.input_dim[1]))(x)

        # Model
        self.encoder = keras.Model(input_img, encoded)
        self.decoder = keras.Model(encoded, decoded)
        self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())


    """
    Compilation of the AutoEncoder.
    Here we specify the method to optimize, the loss function and,
    if we want, some metrics to show together with the loss.
    --------------------------------------------------------------
    string, string, string|array of strings -> 
    --------------------------------------------------------------
    Inputs:
    optimizer: the method to use for optimization. It sould be one
        of tensorflow.keras.optimizers.
    loss: loss function to be optmized. It sould be one of 
        tensorflow.keras.losses.
    metrics: list of metrics to be evaluated by the model during 
        training and testing. It sould be one of
        tensorflow.keras.losses.
    --------------------------------------------------------------
    Outputs:
    No output, only compile the model.
    """
    def compile(self, optimizer='adam', loss='mse', metrics=['mae', 'mape']):
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    """
    Training of the AutoEncoder.
    Here we specify the parameters for our training.
    --------------------------------------------------------------
    ndarray, ndarray, int, int, bool, float, int, 
        float, int, bool -> History
    --------------------------------------------------------------
    Inputs:
    x: input data.
    y: output data (data to be compared and trained).
    batch_size: int or None. Number of samples per epoch (gradient
        update).
    shuffle: boolean whether to shuddle the training data before
        each epoch.
    validation_split: float between 0 and 1. Fraction of the 
        training data to be used as validation data.
    verbose: 'auto', 0, 1 or 2. Differents modes of verbosity when
        the model is trained.
    min_delta: minimum change in the monitored quantity to qualify
        as an improvement.
    patience: number of epochs with no improvement adter wich
        training will be stoped.
    respore_best_weigths: boolean whether to restore model weights
        from the epoch with the best value of the monitored
        quantity
    ---------------------------------------------------------------
    Outputs:
    A History object. See History.history.
    """
    def fit(self, x, y, epochs=100, batch_size=64, shuffle=True, validation_split=0.15, verbose=1, min_delta=3e-6, patience=10, restore_best_weights=True, mask=False):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience,
                                    restore_best_weights=restore_best_weights)
        if mask is not None:
            x[:,mask,0] = 0
            if (x!=y).any():
                y[:,mask,0] = 0
            else:
                y = x
        history = self.autoencoder.fit(x, y,
                            epochs=epochs,
                            shuffle=shuffle,
                            validation_split=validation_split,
                            batch_size=batch_size,
                            callbacks=callback,
                            verbose=verbose
                            )
        return history
