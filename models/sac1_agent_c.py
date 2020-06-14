from spinup.algos.sac1 import core
import tensorflow as tf
import numpy as np
from spinup.algos.sac1.core import get_vars
from numbers import Number
import os, time
import config
import random
import random
seed = 0
data = []
count = 0
N = 1000
part = 0


def get_batch_exp_c(replay_buffer1, replay_buffer2, batch_size = 128):
    global data, count
    if count == 0:
        obs1_1 = replay_buffer1[:, 0]
        obs2_1 = replay_buffer1[:, 1]
        action_1 = replay_buffer1[:, 2]
        reward_1 = replay_buffer1[:, 3]
        d_1 = replay_buffer1[:, 4]
        obs1_2 = replay_buffer2[:, 0]
        obs2_2 = replay_buffer2[:, 1]
        action_2 = replay_buffer2[:, 2]
        reward_2 = replay_buffer2[:, 3]
        d_2 = replay_buffer2[:, 4]
        temp = list(zip(obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2))
        random.shuffle(temp)
        obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2 = zip(*temp)
        data = [obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2]
    elif count < len(data[0]) // batch_size:
        obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2 = data
    elif count >= len(data[0]) // batch_size:
        # print('updated', count, len(data[0]))
        count = 0
        obs1_1 = replay_buffer1[:, 0]
        obs2_1 = replay_buffer1[:, 1]
        action_1 = replay_buffer1[:, 2]
        reward_1 = replay_buffer1[:, 3]
        d_1 = replay_buffer1[:, 4]
        obs1_2 = replay_buffer2[:, 0]
        obs2_2 = replay_buffer2[:, 1]
        action_2 = replay_buffer2[:, 2]
        reward_2 = replay_buffer2[:, 3]
        d_2 = replay_buffer2[:, 4]
        temp = list(zip(obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2))
        random.shuffle(temp)
        obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2 = zip(*temp)
        data = [obs1_1, obs2_1, action_1, reward_1, d_1, obs1_2, obs2_2, action_2, reward_2, d_2]
    print(np.vstack((obs1_1[count*batch_size:(count+1)*batch_size], obs1_2[count*batch_size:(count+1)*batch_size])).shape)
    print(len(obs1_1[count*batch_size:(count+1)*batch_size]), len(obs1_2[count*batch_size:(count+1)*batch_size]))
    print(obs1_1[count*batch_size:(count+1)*batch_size][0])
    exit()

    res = dict(obs1=np.vstack((obs1_1[count*batch_size:(count+1)*batch_size], obs1_2[count*batch_size:(count+1)*batch_size])),
         obs2=np.vstack((obs2_1[count*batch_size:(count+1)*batch_size], obs2_2[count*batch_size:(count+1)*batch_size])),
         acts=np.vstack((action_1[count*batch_size:(count+1)*batch_size], action_2[count*batch_size:(count+1)*batch_size])),
         rews=np.hstack((reward_1[count*batch_size:(count+1)*batch_size], reward_2[count*batch_size:(count+1)*batch_size])),
         done=np.hstack((d_1[count*batch_size:(count+1)*batch_size], d_2[count*batch_size:(count+1)*batch_size])))
    count += 1
    return res

def get_batch_exp_d(replay_buffer1, replay_buffer2, batch_size = 128):
    """ Сэмплирует по порядку из двух буферов в N секциях.
    При этом размер сэмпла одинаковый для обоих буферов.
    """
    global data, count, part

    current_max_1 = (part + 1) * replay_buffer1.shape[0]// 10
    current_max_2 = (part + 1) * replay_buffer1.shape[0]// 10
    indexes_1 = np.random.choice(current_max_1, batch_size, replace=False)
    indexes_2 = np.random.choice(current_max_2, batch_size, replace=False)

    obs1_1, obs2_1, action_1, reward_1, d_1 = sample_from_buffer(replay_buffer1, indexes_1)
    obs1_2, obs2_2, action_2, reward_2, d_2 = sample_from_buffer(replay_buffer2, indexes_2)

    if count == N:
        if part < 9:
            part += 1

        count = 1

    res = dict(obs1=np.vstack((obs1_1, obs1_2)),
               obs2=np.vstack((obs2_1, obs2_2)),
               acts=np.vstack((action_1, action_2)),
               rews=np.hstack((reward_1, reward_2)),
               done=np.hstack((d_1, d_2)))
    count += 1
    return res

def sample_from_buffer(buffer, indexes):
    obs1 = buffer[indexes, 0]
    obs2 = buffer[indexes, 1]
    action = buffer[indexes, 2]
    reward = buffer[indexes, 3]
    done = buffer[indexes, 4]

    return obs1, obs2, action, reward, done

def get_batch_exp_z(replay_buffer1, replay_buffer2, batch_size_1, batch_size_2):
    """ Сэмплирует случайно из двух буферов с разным соотношением элементов 
    (количество элементов для сэмпла свое для каждого буфера)
    """
    # global data, count

    indexes_1 = np.random.choice(replay_buffer1.shape[0], batch_size_1, replace=False)
    indexes_2 = np.random.choice(replay_buffer2.shape[0], batch_size_2, replace=False)

    obs1_1, obs2_1, action_1, reward_1, d_1 = sample_from_buffer(replay_buffer1, indexes_1)
    obs1_2, obs2_2, action_2, reward_2, d_2 = sample_from_buffer(replay_buffer2, indexes_2)

    if batch_size_2 == 0:
        res = dict(obs1=obs1_1,
                   obs2=obs2_1,
                   acts=action_1,
                   rews=reward_1,
                   done=d_1)
    else:
        res = dict(obs1=np.vstack((obs1_1, obs1_2)),
                   obs2=np.vstack((obs2_1, obs2_2)),
                   acts=np.vstack((action_1, action_2)),
                   rews=np.hstack((reward_1, reward_2)),
                   done=np.hstack((d_1, d_2)))
    return res


def train_test(apr, ts_env, env_fn, replay_buffer, replay_buffer2, vae=None, x_train=None, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=5000, epochs=100, replay_size=int(2e6), gamma=0.99, reward_scale=1.0,
         polyak=0.995, lr=5e-4, alpha=0.2, batch_size=250, start_steps=10,
         max_ep_len_train=1000, max_ep_len_test=1000, logger_kwargs=dict(), save_freq=1):

    frames = []
    buffer = []

    tf.set_random_seed(seed)
    np.random.seed(seed)

    print(start_steps)
    epch = 1
    apr.l_ep_ret = -70000
    apr.l_ep_len = 1

    env, test_env = env_fn(3), env_fn(1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, apr.ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, logp_pi2, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, x2_ph, apr.ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, _, logp_pi_, _, _, _, q1_pi_, q2_pi_ = actor_critic(x2_ph, x2_ph, apr.ph, **ac_kwargs)

    # Experience buffer
    # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t total: %d\n') % var_counts)

    ######
    if alpha == 'auto':
        target_entropy = (-np.prod(env.action_space.shape))

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)

        alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))

        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr * 0.1, name='alpha_optimizer')
        train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])
    ######

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi_, q2_pi_)

    # Targets for Q and V regression
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi2)
    q_backup = r_ph + gamma * (1 - d_ph) * v_backup

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
    value_loss = q1_loss + q2_loss

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    if isinstance(alpha, Number):
        step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(alpha),
                    train_pi_op, train_value_op, target_update]
    else:
        step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha,
                    train_pi_op, train_value_op, target_update, train_alpha_op]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    ##############################  save and restore  ############################

    saver = tf.train.Saver()

    #
    # if not os.path.exists(apr.checkpoint_path_r):
    #     os.makedirs(apr.checkpoint_path_r)

    if not os.path.exists(apr.checkpoint_path_wr):
        os.makedirs(apr.checkpoint_path_wr)

    # checkpoint_path_r = apr.checkpoint_path_r

    if apr.is_test or apr.is_restore_train:
        # ckpt = tf.train.get_checkpoint_state(apr.checkpoint_path_wr)
        print("Search ckpt...")
        # if ckpt and ckpt.model_checkpoint_path:
        # saver.restore(sess, ckpt.model_checkpoint_path)
        # print("Model restored.")
        save_path = saver.restore(sess, "content\\model.ckpt")
        print("Model restored in path: %s" % save_path)

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    ##############################  test for train process ############################
    def test_agent(n=25):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            start_pos = test_env.pos[0]
            pit_x = test_env.pit_x
            stump_x = test_env.stump_x
            stairs_x = test_env.stairs_x
            while not (d or (ep_len == max_ep_len_test)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
                if apr.test_render:
                    frames.append(test_env.render(mode='rgb_array'))
                    # test_env.render()
            finish_pos = test_env.pos[0]
            count_pit = 0
            count_stump = 0
            count_stairs = 0
            for pit in pit_x:
                if start_pos < pit < finish_pos:
                    count_pit += 1
            for stump in stump_x:
                if start_pos < stump < finish_pos:
                    count_stump += 1
            for stair in stairs_x:
                if start_pos < stair < finish_pos:
                    count_stairs += 1
            apr.l_ep_ret = int(ep_ret)
            apr.l_ep_len = ep_len
            # print(apr.l_ep_ret)
            return count_pit, count_stump, count_stairs, finish_pos - start_pos, len(pit_x), len(stump_x), len(stairs_x)

    ##############################  train ############################
    start_time = time.time()
    if vae is None and x_train is None:
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    elif vae is not None:
        o, r, d, ep_ret, ep_len = vae.get_data()[0], 0, False, 0, 0
    elif x_train is not None:
        count = 0
        data = x_train[count]
        o, ep_ret, ep_len = data[0], 0, 0
    total_steps = steps_per_epoch * epochs
    print(total_steps)

    test_ep_ret = -10000.0
    test_ep_ret_best = apr.bestr
    n = 50
    min_n = 50
    m = 50
    max_m = 50
    mix_offset = 5000000000  # отвечает за число шагов после которых начинается смешивание буферов
    mix_step = 1 # определяет за какое число шагов смешивание изменяется на единицу
    # Main loop: collect experience in env and update/log each epoch
    batch1 = get_batch_exp_z(replay_buffer, replay_buffer2, n, m)
    batch2 = get_batch_exp_z(replay_buffer, replay_buffer2, n, m)
    for t in range(total_steps):
        for j in range(5000):
            # t = time.time()
            batch = get_batch_exp_z(replay_buffer, replay_buffer2, n, m)
            if t > mix_offset:
                if t % mix_step == 0 and n != min_n:
                    n -= 1
                    m += 1
            # print(time.time()-t)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         apr.ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         }
            # step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha, train_pi_op, train_value_op, target_update]
            outs = sess.run(step_ops, feed_dict)

        # End of epoch wrap-up
        epoch = t
        count_pit, count_stump, count_stairs, way, len_pit_x, len_stump_x, len_stairs = test_agent(1)
        test_ep_ret = apr.l_ep_ret
        print(f'epoch = {epch}, TestEpRet = {test_ep_ret}, Best = {test_ep_ret_best}, пройденный путь = {way},ям - {count_pit}/{len_pit_x}, лестниц - {count_stairs}/{len_stairs}')
        if (count_pit + count_stairs)/(len_pit_x+len_stairs) > 0.8:
            exit()
        epch += 1
        # if test_ep_ret > test_ep_ret_best:
        #     save_path = saver.save(sess, "content\\model.ckpt")
        #     print("Model saved in path: %s" % save_path)
        #     test_ep_ret_best = test_ep_ret
    # np.savez_compressed('replay_{}'.format('ямы'), np.array(buffer))
    # import imageio
    # IMAGE_PATH = 'vae_hole.gif'
    # tf.reset_default_graph()
    # imageio.mimsave(IMAGE_PATH, frames, duration=1.0 / 60.0)
