import sys
import time
import math
import argparse

import numpy as np

import llo.llo as llo
import rocnnet.rocnnet as rcn

prog_description = 'Demo dqn_trainer'

def batch_generate(n, batchsize):
    total = n * batchsize
    return np.random.rand(total)

def observationfit(indata, n_actions):
    out = np.zeros(n_actions)
    n = len(indata)
    for i in range(n):
        out[int(i / n_actions)] = indata[i]
    n = n / n_actions
    for i in range(n_actions):
        out[i] = out[i] / n
    return out

# calculates the circumference distance between A and B assuming
# A and B represent positions on a circle with circumference wrap_size
def wrapdist(A, B, wrap_size):
    within_dist = A - B if A > B else B - A
    edge_dist = min(A + wrap_size - B, B + wrap_size - A)
    return min(within_dist, edge_dist)

def str2bool(opt):
    optstr = opt.lower()
    if optstr in ('yes', 'true', 't', 'y', '1'):
        return True
    elif optstr in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    default_ts = time.time()

    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--seed', dest='seed',
        type=str2bool, nargs='?', const=False, default=True,
        help='Whether to seed or not (default: True)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--n_episodes', dest='n_episodes', type=int, nargs='?', default=250,
        help='Number of episodes (default: 250)')
    parser.add_argument('--max_steps', dest='max_steps', type=int, nargs='?', default=100,
        help='Number of steps per episodes (default: 100)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='pretrained/dqmodel.pbx',
        help='Filename to load pretrained model (default: pretrained/dqmodel.pbx)')
    args = parser.parse_args(args)

    episode_count = args.n_episodes
    max_steps = args.max_steps

    if args.seed:
        print('seeding {}'.format(args.seedval))
        llo.seed(args.seedval)
        np.random.seed(args.seedval)

    n_observations = 32
    n_actions = 29

    hiddens = [
        rcn.get_layer(rcn.tanh, 200),
        rcn.get_layer(rcn.tanh, 200),
        rcn.get_layer(rcn.identity, n_actions)
    ]

    brain = rcn.get_mlp(n_observations, hiddens, 'brain')
    untrained_brain = brain.copy()
    try:
        with open(args.load, 'rb') as f:
            print('loading')
            pretrained_brain = brain.parse_from_string(f.read())
    except:
        pretrained_brain = brain.copy()

    bgd = rcn.get_rms_momentum(
        learning_rate = 0.1,
        discount_factor = 0.5)
    param = rcn.get_dqninfo(
        mini_batch_size = 1,
        store_interval = 1,
        discount_rate = 0.99,
        exploration_period = 0.0)

    untrained_dqn = rcn.DQNTrainer(untrained_brain, bgd, param)
    trained_dqn = rcn.DQNTrainer(brain, bgd, param)
    pretrained_dqn = rcn.DQNTrainer(pretrained_brain, bgd, param)

    err_msg = None
    err_queue_size = 10
    action_dist = int(n_actions / 2)
    n_batch = 1

    error_queue = []
    start = time.time()
    for i in range(episode_count):
        avgreward = 0.0
        observations = batch_generate(n_observations, n_batch)
        expect_out = observationfit(observations, n_actions)
        episode_err = 0.0
        for j in range(max_steps):
            action = trained_dqn.action(observations)
            expect_action = expect_out.argmax()

            # err = [0, 1, 2, 3]
            err = wrapdist(expect_action, action, n_actions)

            reward = 1 - 2.0 * err / action_dist
            avgreward = avgreward + reward

            new_observations = batch_generate(n_observations, n_batch)
            expect_out = observationfit(new_observations, n_actions)

            trained_dqn.store(observations, action, reward, new_observations)
            trained_dqn.train()

            observations = new_observations
            episode_err = episode_err + err / action_dist

        avgreward = avgreward / max_steps
        episode_err = episode_err / max_steps

        error_queue.append(episode_err)
        if len(error_queue) > err_queue_size:
            error_queue = error_queue[1:]

        # allow ~15% decrease in accuracy (15% increase in error) since last episode
        # otherwise declare that we overfitted and quit
        avgerr = 0.0
        for last_error in error_queue:
            avgerr = avgerr + last_error
        avgerr = avgerr / len(error_queue)

        if abs(episode_err - avgerr) > 0.35:
            print("uh oh, we hit a snag, we shouldn't save for this round")
            err_msg = 'overfitting'

        if math.isnan(episode_err):
            raise 'NaN episode error'
        print('episode {} performance: {}% average error, reward: {}'.format(i, episode_err * 100, avgreward))

    print('training time: {} seconds'.format(time.time() - start))

    total_untrained_err = 0.0
    total_trained_err = 0.0
    total_pretrained_err = 0.0
    for j in range(max_steps):
        observations = batch_generate(n_observations, 1)
        expect_out = observationfit(observations, n_actions)

        untrained_action = untrained_dqn.action(observations)
        trained_action = trained_dqn.action(observations)
        pretrained_action = pretrained_dqn.action(observations)

        expect_action = expect_out.argmax()

        untrained_err = wrapdist(expect_action, untrained_action, n_actions)
        trained_err = wrapdist(expect_action, trained_action, n_actions)
        pretrained_err = wrapdist(expect_action, pretrained_action, n_actions)

        total_untrained_err = total_untrained_err + untrained_err / action_dist
        total_trained_err = total_trained_err + trained_err / action_dist
        total_pretrained_err = total_pretrained_err + pretrained_err / action_dist

    total_untrained_err = total_untrained_err / max_steps
    total_trained_err = total_trained_err / max_steps
    total_pretrained_err = total_pretrained_err / max_steps
    print("untrained performance: {}% average error".format(total_untrained_err * 100))
    print("trained performance: {}% average error".format(total_trained_err * 100))
    print("pretrained performance: {}% average error".format(total_pretrained_err * 100))

    # fails if cumulative steps is over threshold=250,
    # and trained is inferior to untrained
    if episode_count * max_steps > 250 and total_untrained_err < total_trained_err:
        err_msg = 'training error rate is wrong'

    if err_msg is None:
        brain.serialize_to_file(trained_dqn.train_out(), args.save)
    else:
        print(err_msg)

if '__main__' == __name__:
    main(sys.argv[1:])
