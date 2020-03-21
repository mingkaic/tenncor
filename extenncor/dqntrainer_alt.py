import math
import random
import numpy as np

import tenncor as tc

def _get_random():
    return random.uniform(0, 1)

# generalize as feedback class
class DQNEnv:
    def __init__(self, dense_model, sess, update_fn,
        gradprocess_fn = None,
        mbatchsize = 32,
        discount_rate = 0.95,
        target_update_rate = 0.01,
        max_exp = 10000,
        explore_period = 1000,
        action_prob = 0.05,
        train_interval = 5,
        store_interval = 5):

        self.mbatchsize = mbatchsize
        self.discount_rate = discount_rate
        self.target_update_rate = target_update_rate
        self.max_exp = max_exp
        self.explore_period = explore_period
        self.action_prob = action_prob
        self.train_interval = train_interval
        self.store_interval = store_interval

        self.sess = sess
        self.source_model = dense_model
        self.target_model = dense_model.deep_clone()

        inshape = list(self.source_model.get_input().shape())
        batchin = [mbatchsize] + inshape
        outshape = [mbatchsize]
        batchout = [mbatchsize] + list(self.source_model.shape())

        # environment interaction
        self.obs = tc.EVariable(inshape, 0, 'training_input')
        act = self.source_model.connect(self.obs)
        self.action_idx = tc.argmax(act)

        # training

        self.src_outmask = tc.EVariable(batchout, 0, 'target_outmask')
        self.nxt_outmask = tc.EVariable(outshape, 0, 'source_outmask')
        self.reward_input = tc.EVariable(outshape, 0, 'reward')

        # forward action score computation
        self.src_obs = tc.EVariable(batchin, 0, 'training_source_input')
        self.src_act = self.source_model.connect(self.src_obs)

        # predicting target future rewards
        self.nxt_obs = tc.EVariable(batchin, 0, 'training_target_input')
        self.nxt_act = self.source_model.connect(self.nxt_obs)
        target_vals = self.nxt_outmask * tc.reduce_max_1d(self.nxt_act, 0)

        future_reward = self.reward_input + discount_rate * target_vals

        prediction_err = tc.reduce_mean(tc.square(
            tc.reduce_sum_1d(self.src_act * self.src_outmask, 0) - future_reward))

        source_vars = self.source_model.get_storage()
        target_vars = self.target_model.get_storage()
        assert(len(source_vars) == len(target_vars))

        source_errs = dict()
        for src_var in source_vars:
            error = tc.derive(prediction_err, src_var)
            if gradprocess_fn is not None:
                error = gradprocess_fn(error)
            source_errs[src_var] = error

        src_updates = dict(update_fn(source_errs))

        self.updates = []
        track_batch = [prediction_err, self.src_act, self.action_idx]
        for nxt_var, src_var in zip(target_vars, source_vars):
            diff = nxt_var - src_updates[src_var]
            assign = tc.assign_sub(nxt_var, target_update_rate * diff)
            track_batch.append(assign)
            self.updates.append(assign)

        self.sess.track(track_batch)
        self.nactions = 0
        self.nstore_called = 0
        self.ntrain_called = 0
        self.experiences = []

    def action(self, input):
        exploration = self._linear_annealing(1., self.nactions)
        self.nactions += 1
        # perform random exploration action
        if _get_random() < exploration:
            return math.floor(_get_random() * self.source_model.shape()[0])

        self.obs.assign(input)
        self.sess.update()
        return  int(self.action_idx.get())

    def store(self, observation, action_idx, reward, new_obs):
        if 0 == self.nstore_called % self.store_interval:
            self.experiences.append((observation, action_idx, reward, new_obs))
            if len(self.experiences) > self.max_exp:
                self.experiences = self.experiences[1:]
        self.nstore_called += 1

    def train(self):
        # extract mini_batch from buffer and backpropagate
        if 0 == (self.ntrain_called % self.train_interval):
            if len(self.experiences) < self.mbatchsize:
                return

            samples = self._random_sample()

            # batch data process
            states = [] # <ninput, batchsize>
            new_states = [] # <ninput, batchsize>
            action_mask = [] # <noutput, batchsize>
            new_states_mask = [] # <batchsize>
            rewards = [] # <batchsize>

            for observation, action_idx, reward, new_obs in samples:
                states.append(observation)
                local_act_mask = [0] * self.source_model.shape()[-1]
                local_act_mask[action_idx] = 1.
                action_mask.append(local_act_mask)
                rewards.append(reward)
                if len(new_obs) == 0:
                    new_states.append([0] * self.source_model.shape()[-1])
                    new_states_mask.append(0)
                else:
                    new_states.append(new_obs)
                    new_states_mask.append(1)

            # enter processed batch data
            self.src_obs.assign(np.array(states))
            self.src_outmask.assign(np.array(action_mask))
            self.nxt_obs.assign(np.array(new_states))
            self.nxt_outmask.assign(np.array(new_states_mask))
            self.reward_input.assign(np.array(rewards))

            self.sess.update_target(self.updates)
        self.ntrain_called += 1

    def _linear_annealing(self, initial_prob, nacts):
        if nacts >= self.explore_period:
            return self.action_prob
        return initial_prob - nacts * (initial_prob - self.action_prob) / self.explore_period

    def _random_sample(self):
        return random.sample(self.experiences, self.mbatchsize)
