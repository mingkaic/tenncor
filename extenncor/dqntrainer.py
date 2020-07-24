import math
import random
import os.path

import numpy as np
from collections.abc import Iterable

import tenncor as tc

import extenncor.trainer_cache as ecache
import extenncor.dqntrainer_pb2 as dqn_pb

_get_random = tc.unif_gen(0, 1)

def get_dqnupdate(update_fn, update_rate):
    def dqnupdate(err, vars):
        halfvars = int(len(vars) / 2)
        src_vars = vars[:halfvars]
        nxt_vars = vars[halfvars:]

        src_updates = dict(update_fn(err, src_vars))

        assigns = []
        for nxt_var, src_var in zip(nxt_vars, src_vars):
            diff = nxt_var - src_updates[src_var]
            assigns.append((nxt_var,
                tc.api.assign_sub(nxt_var, update_rate * diff)))
        return assigns
    return dqnupdate

def get_dqnerror(env, discount_rate):
    def dqnerror(models):
        src_model, nxt_model = tuple(models)

        # forward action score computation
        src_act = src_model.connect(env.src_obs)

        # predicting target future rewards
        nxt_act = nxt_model.connect(env.nxt_obs)
        target_vals = env.nxt_outmask * tc.api.reduce_max_1d(nxt_act, 0)

        future_reward = env.rewards + discount_rate * target_vals

        masked_output_score = tc.api.reduce_sum_1d(src_act * env.src_outmask, 0)
        return tc.api.reduce_mean(tc.api.square(masked_output_score - future_reward))
    return dqnerror

# generalize as feedback class
@ecache.EnvManager.register
class DQNEnv(ecache.EnvManager):
    def __init__(self, src_model, update_fn,
        optimize_cfg = "", max_exp = 30000,
        train_interval = 5, store_interval = 5,
        explore_period = 1000, action_prob = 0.05,
        mbatch_size = 32, discount_rate = 0.95,
        target_update_rate = 0.01, clean_startup = False,
        usecase = '', cachedir = '/tmp',
        ctx = tc.global_context):

        self.max_exp = max_exp
        self.train_interval = train_interval
        self.store_interval = store_interval
        self.explore_period = explore_period
        self.action_prob = action_prob

        def default_init():
            self.actions_executed = 0
            self.ntrain_called = 0
            self.nstore_called = 0
            self.experiences = []

            nxt_model = src_model.deep_clone()

            inshape = list(src_model.get_input().shape())
            batchin = [mbatch_size] + inshape

            # environment interaction
            self.obs = tc.EVariable(inshape, 0, 'obs', ctx=self.ctx)
            self.act_idx = tc.TenncorAPI(self.ctx).argmax(src_model.connect(self.obs))
            self.act_idx.tag("recovery", "act_idx")

            # training
            self.src_obs = tc.EVariable(batchin, 0, 'src_obs', ctx=self.ctx)
            self.nxt_obs = tc.EVariable(batchin, 0, 'nxt_obs', ctx=self.ctx)
            self.src_outmask = tc.EVariable([mbatch_size] + list(src_model.shape()), 1, 'src_outmask', ctx=self.ctx)
            self.nxt_outmask = tc.EVariable([mbatch_size], 1, 'nxt_outmask', ctx=self.ctx)
            self.rewards = tc.EVariable([mbatch_size], 0, 'rewards', ctx=self.ctx)

            self.prediction_err = tc.api.identity(tc.apply_update([src_model, nxt_model],
                get_dqnupdate(update_fn, target_update_rate),
                get_dqnerror(self, discount_rate), ctx=self.ctx))
            self.prediction_err.tag("recovery", "prediction_err")

            tc.optimize(optimize_cfg, self.ctx)

        super().__init__(os.path.join(usecase, 'dqn'),
            ctx=ctx, default_init=default_init,
            clean=clean_startup,  cacheroot=cachedir)
        self.src_shape = self.src_outmask.shape()
        self.mbatch_size = self.rewards.shape()
        if len(self.mbatch_size) > 0:
            self.mbatch_size = self.mbatch_size[0]
        else:
            self.mbatch_size = 1

    def _backup_env(self, fpath: str) -> bool:
        with open(fpath, 'wb') as envfile:
            dqn_env = dqn_pb.DqnEnv()

            dqn_env.actions_executed = self.actions_executed
            dqn_env.ntrain_called = self.ntrain_called
            dqn_env.nstore_called = self.nstore_called
            for obs, act_idx, reward, new_obs in self.experiences:
                exp = dqn_env.experiences.add()
                for ob in obs:
                    exp.obs.append(ob)
                for nob in new_obs:
                    exp.new_obs.append(nob)
                exp.act_idx = act_idx
                exp.reward = reward

            envfile.write(dqn_env.SerializeToString())
            return True
        return False

    def _recover_env(self, fpath: str) -> bool:
        # recover object members from recovered session
        query = tc.Statement(self.ctx.get_actives())
        def safe_recovery(search_q):
            resp = query.find(search_q)
            assert len(resp) == 1, 'search {} returns empty or non-unique result'.format(search_q)
            return resp[0]

        self.obs = tc.to_variable(
            safe_recovery('{ "leaf":{ "label":"obs" } }'), self.ctx)
        self.src_obs = tc.to_variable(
            safe_recovery('{ "leaf":{ "label":"src_obs" } }'), self.ctx)
        self.nxt_obs = tc.to_variable(
            safe_recovery('{ "leaf":{ "label":"nxt_obs" } }'), self.ctx)
        self.src_outmask = tc.to_variable(
            safe_recovery('{ "leaf":{ "label":"src_outmask" } }'), self.ctx)
        self.nxt_outmask = tc.to_variable(
            safe_recovery('{ "leaf":{ "label":"nxt_outmask" } }'), self.ctx)
        self.rewards = tc.to_variable(
            safe_recovery('{ "leaf":{ "label":"rewards" } }'), self.ctx)

        self.act_idx = safe_recovery('''{
            "op":{
                "opname":"ARGMAX",
                "attrs":{
                    "recovery":{
                        "str":"act_idx"
                    }
                }
            }
        }''')
        self.prediction_err = safe_recovery('''{
            "op":{
                "opname":"IDENTITY",
                "attrs":{
                    "recovery":{
                        "str":"prediction_err"
                    }
                }
            }
        }''')

        print('loading environment from "{}"'.format(fpath))
        with open(fpath, 'rb') as envfile:
            dqn_env = dqn_pb.DqnEnv()
            dqn_env.ParseFromString(envfile.read())

            self.actions_executed = dqn_env.actions_executed
            self.ntrain_called = dqn_env.ntrain_called
            self.nstore_called = dqn_env.nstore_called
            self.experiences = [(exp.obs, exp.act_idx, exp.reward, exp.new_obs)
                for exp in dqn_env.experiences]

            print('successfully recovered environment from "{}"'.format(fpath))
            return True

        print('failed environment recovery')
        return False

    def action(self, obs):
        self.actions_executed += 1
        exploration = self._linear_annealing(1.)
        # perform random exploration action
        if _get_random() < exploration:
            return math.floor(_get_random() * self.src_shape[-1])

        self.obs.assign(obs, self.ctx)
        return int(self.act_idx.get())

    def store(self, observation, act_idx, reward, new_obs):
        if 0 == self.nstore_called % self.store_interval:
            self.experiences.append((observation, act_idx, reward, new_obs))
            if len(self.experiences) > self.max_exp:
                self.experiences = self.experiences[1:]
        self.nstore_called += 1

    def train(self):
        if len(self.experiences) < self.mbatch_size:
            return
        # extract mini_batch from buffer and backpropagate
        if 0 == (self.ntrain_called % self.train_interval):
            samples = self._random_sample()

            # batch data process
            states = [] # <ninput, batchsize>
            new_states = [] # <ninput, batchsize>
            action_mask = [] # <noutput, batchsize>
            rewards = [] # <batchsize>

            for observation, act_idx, reward, new_obs in samples:
                assert(len(new_obs) > 0)
                states.append(observation)
                local_act_mask = [0.] * self.src_shape[-1]
                local_act_mask[act_idx] = 1.
                action_mask.append(local_act_mask)
                rewards.append(reward)
                new_states.append(new_obs)

            # enter processed batch data
            states = np.array(states)
            new_states = np.array(new_states)
            action_mask = np.array(action_mask)
            rewards = np.array(rewards)
            self.src_obs.assign(states, self.ctx)
            self.src_outmask.assign(action_mask, self.ctx)
            self.nxt_obs.assign(new_states, self.ctx)
            self.rewards.assign(rewards, self.ctx)

            self.prediction_err.get()
        self.ntrain_called += 1

    def _linear_annealing(self, initial_prob):
        if self.actions_executed >= self.explore_period:
            return self.action_prob
        return initial_prob - self.actions_executed * (initial_prob - self.action_prob) / self.explore_period

    def _random_sample(self):
        return random.sample(self.experiences, self.mbatch_size)
