from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping
from utils import normalize, reverse_normalize

import numpy as np

class Vae:
    def __init__(self, config):
        self.config = config
        self.models, loss = self.create_vae()
        self.vae = self.models['vae']
        self.vae.compile(optimizer='adam', loss=loss)
        self.encoder = self.models['encoder']
        self.decoder = self.models['decoder']


    def create_vae(self):
        original_dim = self.config.ORIGINAL_DIM
        latent_dim = self.config.LATENT_DIM
        structure_encoder = self.config.STRUCTURE_ENCODER
        structure_decoder = self.config.STRUCTURE_DECODER
        models = {}
        inputs = Input(shape=(original_dim,), name='encoder_input')
        for i, layer in enumerate(structure_encoder):
            if i == 0:
                x = Dense(layer[0], activation=layer[1])(inputs)
            else:
                x = Dense(layer[0], activation=layer[1])(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        models["encoder"] = encoder

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        for i, layer in enumerate(structure_decoder):
            if i == 0:
                x = Dense(layer[0], activation=layer[1])(latent_inputs)
            else:
                x = Dense(layer[0], activation=layer[1])(x)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        models["decoder"] = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = models["decoder"](encoder(inputs)[2])
        models["vae"] = Model(inputs, outputs, name='vae')

        def vae_loss(y_true, y_pred):  # combined loss of vae
            reconstruction_loss = binary_crossentropy(inputs, outputs)
            reconstruction_loss *= original_dim
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            print(y_true)
            # classification_loss = categorical_crossentropy(y_true,classifier_outputs)
            # classification_loss *= classification_loss_factor
            return K.mean(reconstruction_loss + kl_loss)  # + classification_loss)
        return models, vae_loss
    def fit_vae(self, x_train):
        x_train, self.minimums, self.maximums = normalize(x_train)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.vae.fit(x_train, x_train, epochs=self.config.EPOCHS, batch_size=self.config.BATCH_SIZE,
                     validation_split=self.config.VALIDATION_SPLIT, callbacks=[early_stopping])
    def get_data(self):
        z_sample = np.random.randn(1, self.config.LATENT_DIM)
        X_batch = self.decoder.predict(z_sample)
        X_batch = reverse_normalize(X_batch, self.minimums, self.maximums)[0]
        obs1 = X_batch[:24]
        obs2 = X_batch[24:48]
        acts = X_batch[48:52]
        rews = X_batch[52]
        d = False
        return obs1, obs2, acts, rews, d
    def testing(self, x_train):
        x_train, minimums, maximums = normalize(x_train)
        X_gen = np.empty(shape=[0, self.config.ORIGINAL_DIM])  # array to store generated samples

        for _ in range(500):  # generate samples in batches
            z_sample = np.random.randn(self.config.BATCH_SIZE, self.config.LATENT_DIM)  # sampling latent vector
            X_batch = self.decoder.predict(z_sample)  # generate corresponding samples
            X_gen = np.append(X_gen, X_batch, axis=0)  # add dreams to the collection

        print(minimums, maximums)
        print('до обратного масштабирования')
        print(
            'np.min(X_gen) = {}, np.max(X_gen) = {}, np.mean(X_gen) = {}, np.median(X_gen)) = {}'.format(np.min(X_gen),
                                                                                                         np.max(X_gen),
                                                                                                         np.mean(X_gen),
                                                                                                         np.median(
                                                                                                             X_gen)))
        print('np.min(x_train) = {}, np.max(x_train) = {}, np.mean(x_train) = {}, np.median(x_train)) = {}'.format(
            np.min(x_train), np.max(x_train), np.mean(x_train), np.median(x_train)))
        X_gen = reverse_normalize(X_gen, minimums, maximums)
        x_train = reverse_normalize(x_train, minimums, maximums)
        print('после обратного масштабирования')
        print(
            'np.min(X_gen) = {}, np.max(X_gen) = {}, np.mean(X_gen) = {}, np.median(X_gen)) = {}'.format(np.min(X_gen),
                                                                                                         np.max(X_gen),
                                                                                                         np.mean(X_gen),
                                                                                                         np.median(
                                                                                                             X_gen)))
        print('np.min(x_train) = {}, np.max(x_train) = {}, np.mean(x_train) = {}, np.median(x_train)) = {}'.format(
            np.min(x_train), np.max(x_train), np.mean(x_train), np.median(x_train)))

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# CVAE model = encoder + decoder + classifier

def create_and_fit_vae(original_dim, latent_dim, structure_encoder, structure_decoder, fn_activation='relu'):
    models = {}

    # build encoder model
    inputs = Input(shape=(original_dim,), name='encoder_input')
    for i, neurons in enumerate(structure_encoder):
        if i==0:
            x = Dense(neurons, activation=fn_activation)(inputs)
        else:
            x = Dense(neurons, activation=fn_activation)(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    models["encoder"] = encoder

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    for i, neurons in enumerate(structure_decoder):
        if i==0:
            x = Dense(neurons, activation=fn_activation)(latent_inputs)
        else:
            x = Dense(neurons * 2, activation=fn_activation)(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    models["decoder"] = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = models["decoder"](encoder(inputs)[2])
    models["vae"] = Model(inputs, outputs, name='vae')

    def vae_loss(y_true, y_pred):  # combined loss of vae
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        print(y_true)
        # classification_loss = categorical_crossentropy(y_true,classifier_outputs)
        # classification_loss *= classification_loss_factor
        return K.mean(reconstruction_loss + kl_loss)  # + classification_loss)

    return models, vae_loss


