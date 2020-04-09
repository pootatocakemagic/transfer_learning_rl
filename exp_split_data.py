from models.dense import DenseNet
from models.sac1 import Sac1, Sac2
from models.vae import Vae
from utils import *
import config
from lib.aparam import aparam
from lib.env import Wrapper, BWg, BWpit, BWstu
from lib.replay_buffer import ReplayBuffer


class new_vae(Vae):

    def get_data(self):
        z_sample = np.random.randn(1, self.config.LATENT_DIM)
        X_batch = self.decoder.predict(z_sample)
        return reverse_normalize(X_batch, self.minimums, self.maximums)[0]

    def testing(self, x_train):
        x_train, self.minimums, self.maximums = normalize(x_train)
        X_gen = np.empty(shape=[0, self.config.ORIGINAL_DIM])  # array to store generated samples

        for _ in range(500):  # generate samples in batches
            z_sample = np.random.randn(self.config.BATCH_SIZE, self.config.LATENT_DIM)  # sampling latent vector
            X_batch = self.decoder.predict(z_sample)  # generate corresponding samples
            X_gen = np.append(X_gen, X_batch, axis=0)  # add dreams to the collection
        print(self.minimums, self.maximums)
        print('до обратного масштабирования')
        print('X_gen:')
        print(f'obs1: min = {np.min(X_gen)}, max = {np.max(X_gen)}, mean = {np.mean(X_gen)}; ')
        print('X_train:')
        print(f'obs1: min = {np.min(x_train)}, max = {np.max(x_train)}, mean = {np.mean(x_train)}; ')
        X_gen = reverse_normalize(X_gen, self.minimums, self.maximums)
        x_train = reverse_normalize(x_train, self.minimums, self.maximums)
        print('после обратного масштабирования')
        print('X_gen:')
        print(f'obs1: min = {np.min(X_gen)}, max = {np.max(X_gen)}, mean = {np.mean(X_gen)}; ')
        print('X_train:')
        print(
            f'obs1: min = {np.min(x_train)}, max = {np.max(x_train)}, mean = {np.mean(x_train)}; ')


ENV = 'трава' # возможные значения 'трава' 'ямы' 'пни'
TRAIN_VAE = False
TRAIN_DENSE = False

vae = new_vae(config)
x_train = np.load('replay_{}.npz'.format(ENV))['arr_0']
apr = aparam()
replay_buffer = ReplayBuffer(24, 4, int(2e6))
x_train_vae = x_train[:, :24]
x_train_dense = np.hstack((x_train[:, :24],x_train[:, 48:52]))
y_train_dense = np.hstack((x_train[:, 24:48], x_train[:, 52].reshape(x_train.shape[0], 1)))
if TRAIN_VAE:
    vae.fit_vae(x_train_vae)
    vae.testing(x_train_vae)
    vae.save_model()
else:
    vae.load_model()
    vae.testing(x_train_vae)
if ENV == 'трава':
    env3 = Wrapper(BWg(), apr, 3)  # трава
    env1 = BWg()
    ts_env = BWg()
elif ENV == 'ямы':
    env3 = Wrapper(BWpit(), apr, 3)  # ямы
    env1 = BWpit()
    ts_env = BWpit()
elif ENV == 'пни':
    env3 = Wrapper(BWstu(), apr, 3)  # пни
    env1 = BWstu()
    ts_env = BWstu()
else:
    print('задайте корректное окружение(ENV) возможные - трава, ямы, пни')
    exit()

apr.is_restore_train = True
apr.is_test = True
sac1 = Sac1(apr, ts_env, lambda n: env3 if n == 3 else env1, replay_buffer)
data = vae.get_data()
action = sac1.get_action(data)
print(sac1.get_action(data))
densenet = DenseNet()
if TRAIN_DENSE:
    densenet.fit_model(x_train_dense, y_train_dense)
    densenet.save_model()
else:
    densenet.load_model()
print(data.shape)
print(data.reshape(1, -1).shape)
print(action.shape)
print(x_train_dense.shape)
print(densenet.get_data(np.hstack((data.reshape(1, -1), action.reshape(1, -1)))))
apr.is_restore_train = False
apr.is_test = False
final_sac1 = Sac2(apr, ts_env, lambda n: env3 if n == 3 else env1, replay_buffer, vae, densenet, sac1)
