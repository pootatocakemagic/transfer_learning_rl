from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras import regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from utils import normalize_vae, reverse_normalize_vae

import numpy as np

class Vae:
    def __init__(self, config):
        self.config = config
        self.models, loss = self.create_vae()
        self.vae = self.models['vae']
        self.vae.compile(optimizer='adam', loss=loss)
        self.encoder = self.models['encoder']
        self.decoder = self.models['decoder']

    def sampling(self, args):
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


    def create_vae(self):
        original_dim = self.config.ORIGINAL_DIM
        latent_dim = self.config.LATENT_DIM
        structure_encoder = self.config.STRUCTURE_ENCODER
        structure_decoder = self.config.STRUCTURE_DECODER
        models = {}
        L2 = 0.001
        inputs = Input(shape=(original_dim,), name='encoder_input')
        for i, layer in enumerate(structure_encoder):
            if i == 0:
                x = Dense(layer[0], activation=layer[1], kernel_regularizer=regularizers.l2(L2),bias_regularizer=regularizers.l2(L2))(inputs)
            else:
                x = Dense(layer[0], activation=layer[1], kernel_regularizer=regularizers.l2(L2),bias_regularizer=regularizers.l2(L2))(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        models["encoder"] = encoder

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        for i, layer in enumerate(structure_decoder):
            if i == 0:
                x = Dense(layer[0], activation=layer[1], kernel_regularizer=regularizers.l2(L2),bias_regularizer=regularizers.l2(L2))(latent_inputs)
            else:
                x = Dense(layer[0], activation=layer[1], kernel_regularizer=regularizers.l2(L2),bias_regularizer=regularizers.l2(L2))(x)
        outputs = Dense(original_dim, activation=self.config.OUTPUT_FN_ACTIVATION)(x)

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
        x_train, self.minimums, self.maximums = normalize_vae(x_train.copy())
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.vae.fit(x_train, x_train, epochs=self.config.EPOCHS, batch_size=self.config.BATCH_SIZE,
                     validation_split=self.config.VALIDATION_SPLIT, callbacks=[early_stopping])
        # self.vae.fit(x_train, x_train, epochs=self.config.EPOCHS, batch_size=self.config.BATCH_SIZE,
        #              validation_split=self.config.VALIDATION_SPLIT)

    def save_model(self):
        self.decoder.save_weights("model_vae.h5")

    def load_model(self):
        self.decoder.load_weights("model_vae.h5")

    def get_data(self):
        z_sample = np.random.randn(1, self.config.LATENT_DIM)
        X_batch = self.decoder.predict(z_sample)
        X_batch = reverse_normalize_vae(X_batch, self.minimums, self.maximums)[0]
        obs1 = X_batch[:24]
        obs2 = X_batch[24:48]
        acts = X_batch[48:52]
        rews = X_batch[52]
        d = False
        return obs1, obs2, acts, rews, d
    

    def testing(self, x_train):
        x_train, minimums, maximums = normalize_vae(x_train)
        X_gen = np.empty(shape=[0, self.config.ORIGINAL_DIM])  # array to store generated samples

        for _ in range(500):  # generate samples in batches
            z_sample = np.random.randn(self.config.BATCH_SIZE, self.config.LATENT_DIM)  # sampling latent vector
            X_batch = self.decoder.predict(z_sample)  # generate corresponding samples
            X_gen = np.append(X_gen, X_batch, axis=0)  # add dreams to the collection

        print(minimums, maximums)
        print('до обратного масштабирования')
        print('X_gen:')
        print(f'obs1: min = {np.min(X_gen[:, :24])}, max = {np.max(X_gen[:, :24])}, mean = {np.mean(X_gen[:, :24])}; '
              f'\nobs2: min = {np.min(X_gen[:, 24:48])}, max = {np.max(X_gen[:, 24:48])}, mean = {np.mean(X_gen[:, 24:48])}; '
              f'\nacts: min = {np.min(X_gen[:, 48:52])}, max = {np.max(X_gen[:, 48:52])}, mean = {np.mean(X_gen[:, 48:52])}; '
              f'\nrews: min = {np.min(X_gen[:, 52])}, max = {np.max(X_gen[:, 52])}, mean = {np.mean(X_gen[:, 52])};'
              f'\nd: min = {np.min(x_train[:, 53])}, max = {np.max(x_train[:, 53])}, mean = {np.mean(x_train[:, 53])};')
        print('X_train:')
        print(f'obs1: min = {np.min(x_train[:, :24])}, max = {np.max(x_train[:, :24])}, mean = {np.mean(x_train[:, :24])}; '
              f'\nobs2: min = {np.min(x_train[:, 24:48])}, max = {np.max(x_train[:, 24:48])}, mean = {np.mean(x_train[:, 24:48])}; '
              f'\nacts: min = {np.min(x_train[:, 48:52])}, max = {np.max(x_train[:, 48:52])}, mean = {np.mean(x_train[:, 48:52])}; '
              f'\nrews: min = {np.min(x_train[:, 52])}, max = {np.max(x_train[:, 52])}, mean = {np.mean(x_train[:, 52])};'
              f'\nd: min = {np.min(x_train[:, 53])}, max = {np.max(x_train[:, 53])}, mean = {np.mean(x_train[:, 53])};')
        X_gen = reverse_normalize_vae(X_gen, minimums, maximums)
        x_train = reverse_normalize_vae(x_train, minimums, maximums)
        print('после обратного масштабирования')
        print('X_gen:')
        print(f'obs1: min = {np.min(X_gen[:, :24])}, max = {np.max(X_gen[:, :24])}, mean = {np.mean(X_gen[:, :24])}; '
              f'\nobs2: min = {np.min(X_gen[:, 24:48])}, max = {np.max(X_gen[:, 24:48])}, mean = {np.mean(X_gen[:, 24:48])}; '
              f'\nacts: min = {np.min(X_gen[:, 48:52])}, max = {np.max(X_gen[:, 48:52])}, mean = {np.mean(X_gen[:, 48:52])}; '
              f'\nrews: min = {np.min(X_gen[:, 52])}, max = {np.max(X_gen[:, 52])}, mean = {np.mean(X_gen[:, 52])};'
              f'\nd: min = {np.min(x_train[:, 53])}, max = {np.max(x_train[:, 53])}, mean = {np.mean(x_train[:, 53])};')
        print('X_train:')
        print(
            f'obs1: min = {np.min(x_train[:, :24])}, max = {np.max(x_train[:, :24])}, mean = {np.mean(x_train[:, :24])}; '
            f'\nobs2: min = {np.min(x_train[:, 24:48])}, max = {np.max(x_train[:, 24:48])}, mean = {np.mean(x_train[:, 24:48])}; '
            f'\nacts: min = {np.min(x_train[:, 48:52])}, max = {np.max(x_train[:, 48:52])}, mean = {np.mean(x_train[:, 48:52])}; '
            f'\nrews: min = {np.min(x_train[:, 52])}, max = {np.max(x_train[:, 52])}, mean = {np.mean(x_train[:, 52])};'
            f'\nd: min = {np.min(x_train[:, 53])}, max = {np.max(x_train[:, 53])}, mean = {np.mean(x_train[:, 53])};')
