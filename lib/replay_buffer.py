import numpy as np


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def export_np(self, batch_size=1000):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def sample_batch2(self, replay_buffer2, batch_size=32, batch_size2=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs2 = np.random.randint(0, replay_buffer2.size, size=batch_size2)

        return dict(obs1=np.vstack((replay_buffer2.obs1_buf[idxs2], self.obs1_buf[idxs])),
                    obs2=np.vstack((replay_buffer2.obs2_buf[idxs2], self.obs2_buf[idxs])),
                    acts=np.vstack((replay_buffer2.acts_buf[idxs2], self.acts_buf[idxs])),
                    rews=np.hstack((replay_buffer2.rews_buf[idxs2], self.rews_buf[idxs])),
                    done=np.hstack((replay_buffer2.done_buf[idxs2], self.done_buf[idxs])))