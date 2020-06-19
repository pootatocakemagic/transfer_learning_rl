from lib.aparam import aparam
from lib.replay_buffer import ReplayBuffer
from spinup.utils.run_utils import setup_logger_kwargs
from lib.env import Wrapper, BWg, BWpit, BWstu, BWstapit
from models.sac1 import sac1
from models.sac1_vae_agents import sac1 as sac
from spinup.algos.sac1 import core
import numpy as np
from models.vae import Vae
import config
import pickle
GET_REPLAY_BUFFER = False
FIT_VAE_TEST_AGENT = True
ENV = 'ямы' # возможные значения 'трава' 'ямы' 'пни'
# x_train = np.load('replay_{}.npz'.format("трава228"), allow_pickle=True)['arr_0']
# print(x_train[0][0])
# exit()
if GET_REPLAY_BUFFER:

      apr = aparam()
      replay_buffer = ReplayBuffer(24, 4, int(2e6))
      logger_kwargs = setup_logger_kwargs(apr.exp_name, apr.seed)
      if ENV == 'трава':
            env3 = Wrapper(BWg(), apr, 3)     # трава
            env1 = BWg()
            ts_env = BWg()
      elif ENV == 'ямы':
            env3 = Wrapper(BWpit(), apr, 3)   # ямы
            env1 = BWpit()
            ts_env = BWpit()
      elif ENV == 'пни':
            env3 = Wrapper(BWstu(), apr, 3)     # пни
            env1 = BWstu()
            ts_env = BWstu()
      elif ENV == 'Лестницы':
            env3 = Wrapper(BWstapit(), apr, 3)  # лестницы и ямы
            env1 = BWstapit()
            ts_env = BWstapit()
      else:
            print('задайте корректное окружение(ENV) возможные - трава, ямы, пни')
            exit()
      sac1(apr, ts_env, lambda n: env3 if n == 3 else env1, replay_buffer, ENV, actor_critic=core.mlp_actor_critic,
            ac_kwargs=dict(hidden_sizes=[400, 300]),
            gamma=apr.gamma, seed=apr.seed, epochs=apr.epochs, alpha=apr.alpha,
            logger_kwargs=logger_kwargs, lr=apr.lr, reward_scale=apr.reward_scale, start_steps = apr.start_steps,
            max_ep_len_train=apr.max_ep_len_train, max_ep_len_test=apr.max_ep_len_test)

      # ep = 100
      # print(replay_buffer.size)
      # print(replay_buffer.max_size)
      # print(replay_buffer.obs1_buf[ep])
      # print(replay_buffer.obs2_buf[ep])
      # print(replay_buffer.acts_buf[ep])
      # print(replay_buffer.rews_buf[ep])
      # print(replay_buffer.done_buf[ep])
      # np_skill = np.hstack((replay_buffer.obs1_buf,replay_buffer.obs2_buf, replay_buffer.acts_buf, replay_buffer.rews_buf.reshape(replay_buffer.rews_buf.shape[0], 1)))
      # np_skill = np_skill[0:replay_buffer.size]
      # print(np_skill[ep])
      # np.savez_compressed('replay_{}'.format(ENV), np_skill)

if FIT_VAE_TEST_AGENT:
      # vae1 = Vae(config)
      # # x_train = np.load('replay_{}.npz'.format(ENV))['arr_0']
      # x_train = np.load('replay_лестницы.npz', allow_pickle=True)['arr_0']
      # y = np.array([np.array(xi) for xi in x_train[:, 0]])
      # for i in range(1, 5):
      #       if i < 3:
      #             temp = np.array([np.array(xi) for xi in x_train[:, i]])
      #       else:
      #             temp = np.array([np.array(xi) for xi in x_train[:, i]]).reshape(-1, 1)
      #       y = np.concatenate([y, temp], axis=1)
      # vae1.fit_vae(y)
      #
      # vae1.testing(y)
      # vae1.save_model('vae1.h5')
      # vae2 = Vae(config)
      # # x_train = np.load('replay_{}.npz'.format(ENV))['arr_0']
      # x_train = np.load('replay_ямы.npz', allow_pickle=True)['arr_0']
      # y = np.array([np.array(xi) for xi in x_train[:, 0]])
      # for i in range(1, 5):
      #       if i < 3:
      #             temp = np.array([np.array(xi) for xi in x_train[:, i]])
      #       else:
      #             temp = np.array([np.array(xi) for xi in x_train[:, i]]).reshape(-1, 1)
      #       y = np.concatenate([y, temp], axis=1)
      # vae2.fit_vae(y)
      # vae2.testing(y)
      # vae2.save_model('vae2.h5')
      apr = aparam()
      # replay_buffer = ReplayBuffer(24, 4, int(2e6))
      logger_kwargs = setup_logger_kwargs(apr.exp_name, apr.seed)
      if ENV == 'трава':
            env3 = Wrapper(BWg(), apr, 3)     # трава
            env1 = BWg()
            ts_env = BWg()
      elif ENV == 'ямы':
            env3 = Wrapper(BWpit(), apr, 3)   # ямы
            env1 = BWpit()
            ts_env = BWpit()
      elif ENV == 'пни':
            env3 = Wrapper(BWstu(), apr, 3)     # пни
            env1 = BWstu()
            ts_env = BWstu()
      else:
            print('задайте корректное окружение(ENV) возможные - трава, ямы, пни')
            exit()
      env3 = Wrapper(BWstapit(), apr, 3)  # лестницы и ямы
      env1 = BWstapit()
      ts_env = BWstapit()
      # vae2 = [1,2]
      sac(apr, ts_env, lambda n: env3 if n == 3 else env1, x_train=None, actor_critic=core.mlp_actor_critic,
           ac_kwargs=dict(hidden_sizes=[400, 300]),
           gamma=apr.gamma, seed=apr.seed, epochs=apr.epochs, alpha=apr.alpha,
           logger_kwargs=logger_kwargs, lr=apr.lr, reward_scale=apr.reward_scale, start_steps=apr.start_steps,
           max_ep_len_train=apr.max_ep_len_train, max_ep_len_test=apr.max_ep_len_test)
