class aparam:
    def __init__(self):
        self.is_restore_train = False
        self.is_test = False
        self.test_render = True
        self.max_ep_len_test = 2000
        self.max_ep_len_train = 400
        self.hid = 300
        self.l = 1
        self.gamma = 0.99
        self.lr = 1e-4
        self.seed = 0
        self.epochs = 30 # 100000
        self.alpha = 0.1
        self.reward_scale = 5.0
        self.act_noise = 0.3
        self.obs_noise = 0.0
        self.exp_name = 'base'
        self.start_steps = 1
        self.checkpoint_path_wr = 'C:\\Users\\Администратор\\DRL\\saved_models\\A_sac1_BipedalWalker-v2_3e6_1\\A_sac1_BipedalWalker-v2_3e6_1\\A_sac1_BipedalWalker-v2_3e6_1_s0'
        self.bestr = 0