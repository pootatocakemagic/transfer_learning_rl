from lib.aparam import aparam
from lib.replay_buffer import ReplayBuffer
from spinup.utils.run_utils import setup_logger_kwargs
from lib.env import Wrapper, BWg, BWpit, BWstu, BWstapit
from models.sac1_agent_c import train_test
from spinup.algos.sac1 import core
import numpy as np
from models.vae import Vae
import config
import pickle
import random
GET_REPLAY_BUFFER = True
FIT_VAE_TEST_AGENT = False
apr = aparam()
ENV = 'ямы' # возможные значения 'трава' 'ямы' 'пни'
# x_train = np.load('replay_{}.npz'.format("трава228"), allow_pickle=True)['arr_0']
# print(x_train[0][0])
# exit()
replay_buffer = np.load('replay_лестницы.npz'.format(ENV), allow_pickle=True)['arr_0']
replay_buffer2 = np.load('replay_ямы.npz'.format(ENV), allow_pickle=True)['arr_0']
logger_kwargs = setup_logger_kwargs(apr.exp_name, apr.seed)
env3 = Wrapper(BWstapit(), apr, 3)     # лестницы и ямы
env1 = BWstapit()
ts_env = BWstapit()
train_test(apr, ts_env, lambda n: env3 if n == 3 else env1, replay_buffer, replay_buffer2, actor_critic=core.mlp_actor_critic,
            ac_kwargs=dict(hidden_sizes=[400, 300]),
            gamma=apr.gamma, seed=apr.seed, epochs=apr.epochs, alpha=apr.alpha,
            logger_kwargs=logger_kwargs, lr=apr.lr, reward_scale=apr.reward_scale, start_steps=apr.start_steps,
            max_ep_len_train=apr.max_ep_len_train, max_ep_len_test=apr.max_ep_len_test)
# if GET_REPLAY_BUFFER:
#
#       apr = aparam()
#       replay_buffer = ReplayBuffer(24, 4, int(2e6))
#       logger_kwargs = setup_logger_kwargs(apr.exp_name, apr.seed)
#       if ENV == 'трава':
#             env3 = Wrapper(BWg(), apr, 3)     # трава
#             env1 = BWg()
#             ts_env = BWg()
#       elif ENV == 'ямы':
#             env3 = Wrapper(BWpit(), apr, 3)   # ямы
#             env1 = BWpit()
#             ts_env = BWpit()
#       elif ENV == 'пни':
#             env3 = Wrapper(BWstu(), apr, 3)     # пни
#             env1 = BWstu()
#             ts_env = BWstu()
#       else:
#             print('задайте корректное окружение(ENV) возможные - трава, ямы, пни')
#             exit()
#       sac1(apr, ts_env, lambda n: env3 if n == 3 else env1, replay_buffer, actor_critic=core.mlp_actor_critic,
#             ac_kwargs=dict(hidden_sizes=[400, 300]),
#             gamma=apr.gamma, seed=apr.seed, epochs=apr.epochs, alpha=apr.alpha,
#             logger_kwargs=logger_kwargs, lr=apr.lr, reward_scale=apr.reward_scale, start_steps=apr.start_steps,
#             max_ep_len_train=apr.max_ep_len_train, max_ep_len_test=apr.max_ep_len_test)
#
#       # ep = 100
#       # print(replay_buffer.size)
#       # print(replay_buffer.max_size)
#       # print(replay_buffer.obs1_buf[ep])
#       # print(replay_buffer.obs2_buf[ep])
#       # print(replay_buffer.acts_buf[ep])
#       # print(replay_buffer.rews_buf[ep])
#       # print(replay_buffer.done_buf[ep])
#       # np_skill = np.hstack((replay_buffer.obs1_buf,replay_buffer.obs2_buf, replay_buffer.acts_buf, replay_buffer.rews_buf.reshape(replay_buffer.rews_buf.shape[0], 1)))
#       # np_skill = np_skill[0:replay_buffer.size]
#       # print(np_skill[ep])
#       # np.savez_compressed('replay_{}'.format(ENV), np_skill)
#
# if FIT_VAE_TEST_AGENT:
#       vae = Vae(config)
#       # x_train = np.load('replay_{}.npz'.format(ENV))['arr_0']
#       x_train = np.load('replay_{}.npz'.format(ENV), allow_pickle=True)['arr_0']
#       y = np.array([np.array(xi) for xi in x_train[:, 0]])
#       for i in range(1, 5):
#             if i < 3:
#                   temp = np.array([np.array(xi) for xi in x_train[:, i]])
#             else:
#                   temp = np.array([np.array(xi) for xi in x_train[:, i]]).reshape(-1, 1)
#             y = np.concatenate([y, temp], axis=1)
#       vae.fit_vae(y)
#
#       vae.testing(y)
#       apr = aparam()
#       replay_buffer = ReplayBuffer(24, 4, int(2e6))
#       logger_kwargs = setup_logger_kwargs(apr.exp_name, apr.seed)
#       if ENV == 'трава':
#             env3 = Wrapper(BWg(), apr, 3)     # трава
#             env1 = BWg()
#             ts_env = BWg()
#       elif ENV == 'ямы':
#             env3 = Wrapper(BWpit(), apr, 3)   # ямы
#             env1 = BWpit()
#             ts_env = BWpit()
#       elif ENV == 'пни':
#             env3 = Wrapper(BWstu(), apr, 3)     # пни
#             env1 = BWstu()
#             ts_env = BWstu()
#       else:
#             print('задайте корректное окружение(ENV) возможные - трава, ямы, пни')
#             exit()
#       ts_env = BWg()
#       sac1(apr, ts_env, lambda n: env3 if n == 3 else env1, replay_buffer, x_train=None, vae=vae, actor_critic=core.mlp_actor_critic,
#            ac_kwargs=dict(hidden_sizes=[400, 300]),
#            gamma=apr.gamma, seed=apr.seed, epochs=apr.epochs, alpha=apr.alpha,
#            logger_kwargs=logger_kwargs, lr=apr.lr, reward_scale=apr.reward_scale, start_steps=apr.start_steps,
#            max_ep_len_train=apr.max_ep_len_train, max_ep_len_test=apr.max_ep_len_test)
