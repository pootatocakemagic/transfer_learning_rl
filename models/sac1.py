from spinup.algos.sac1 import core
import tensorflow as tf
import numpy as np
from spinup.algos.sac1.core import get_vars
from numbers import Number
import os, time
import config

class Sac1:
    def __init__(self, config):
        self.config = config

    def init_model(self):
        pass
    def train(self):
        pass
    def test(self):
        pass

def sac1(apr, ts_env, env_fn, replay_buffer, vae=None, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=5000, epochs=100, replay_size=int(2e6), gamma=0.99, reward_scale=1.0,
         polyak=0.995, lr=5e-4, alpha=0.2, batch_size=250, start_steps=10,
         max_ep_len_train=1000, max_ep_len_test=1000, logger_kwargs=dict(), save_freq=1):
    #   '''
    # def sac1(apr,ts_env, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
    #          steps_per_epoch=5000, epochs=100, replay_size=int(2e6), gamma=0.99, reward_scale=1.0,
    #          polyak=0.995, lr=5e-4, alpha=0.2, batch_size=250, start_steps=10000,
    #          max_ep_len_train=1000, max_ep_len_test=1000, logger_kwargs=dict(), save_freq=1):
    #   '''

    # if not apr.is_test:
    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())
    frames = []

    tf.set_random_seed(seed)
    np.random.seed(seed)

    print(start_steps)

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

    ##############################  test  ############################

    if apr.is_test:
        # test_env = gym.make(a_env)
        test_env = ts_env
        # test_env = BWg()
        ave_ep_ret = 0
        for j in range(start_steps):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len_test)):
                action = get_action(o, True)
                o, r, d, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
                if apr.test_render:
                    frames.append(test_env.render(mode='rgb_array'))
                    # test_env.render()
            ave_ep_ret = (j * ave_ep_ret + ep_ret) / (j + 1)
            print('ep_len', ep_len, 'ep_ret:', ep_ret, 'ave_ep_ret:', ave_ep_ret, '--- {}  /'.format(j + 1),
                  start_steps)
        return

    ##############################  train  ############################

    def test_agent(n=25):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len_test)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
                if apr.test_render:
                    frames.append(test_env.render(mode='rgb_array'))
                    # test_env.render()
            apr.l_ep_ret = int(ep_ret)
            apr.l_ep_len = ep_len
            # print(apr.l_ep_ret)

    # --------------------------------------------

    start_time = time.time()
    if vae is None:
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    else:
        o, r, d, ep_ret, ep_len = vae.get_data()[0], 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    test_ep_ret = -10000.0
    test_ep_ret_best = apr.bestr

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()
        if vae is None:
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            replay_buffer.store(o, a, r, o2, d)
            o = o2
        else:
            o, o2, a, r, d = vae.get_data()
            ep_ret += r
            ep_len += 1
            replay_buffer.store(o, a, r, o2, d)

        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len_train):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             apr.ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             }
                # step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha, train_pi_op, train_value_op, target_update]
                outs = sess.run(step_ops, feed_dict)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            test_agent(1)
            test_ep_ret = apr.l_ep_ret
            print('TestEpRet', test_ep_ret, 'Best:', test_ep_ret_best)
            if test_ep_ret > test_ep_ret_best:
                save_path = saver.save(sess, "content\\model.ckpt")
                print("Model saved in path: %s" % save_path)
                test_ep_ret_best = test_ep_ret