'''Module that contains utilities for use by TRPO class.'''
from collections import defaultdict
from itertools import chain, count

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - ob: (np.ndarray) observations
        - rew: (numpy float) rewards (if gail is used it is the predicted reward)
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode -> True if first timestep of an episode)
        - ac: (np.ndarray) actions
        - prevac: (np.ndarray) previous actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (
        gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    obs_shape = env.observation_space.shape
    # not used, just so we have the datatype
    action = env.action_space.sample()
    action = np.reshape(action, (-1, *env.action_space.shape))
    observation = env.reset().reshape((-1, *obs_shape))

    # ep_ret  # return in current episode
    # ep_true_ret # true return (Applies if GAIL)
    # ep_len  # len of current episode
    current = defaultdict(lambda: 0)
    current_it_len = 0  # len of current iteration
    # ep_rets  # returns of completed episodes in this segment
    # ep_true_rets # true returns of completed episodes in this segment (GAIL)
    # ep_lens  # Episode lengths
    segments = defaultdict(list)

    # Initialize history arrays
    data = defaultdict(list)
    states = policy.initial_state
    done = True  # marks if we're on first timestep of an episode

    for step in count():
        prevac = action
        action, vpred, states, _ = policy.step(observation, states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                'total_timestep': current_it_len,
                "nextvpred": vpred[0] * (1 - done),
                **{k: np.array(d) for k, d in data.items()},
                **segments
            }
            data.clear()
            _, vpred, _, _ = policy.step(observation)
            segments.clear()
            # Reset current iteration length
            current_it_len = 0
        data['ob'].extend(observation)
        data['vpred'].extend(vpred)
        data['ac'].extend(action)
        data['prevac'].extend(prevac)

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(
                action, env.action_space.low, env.action_space.high)

        observation, true_rew, done, _ = env.step(clipped_action)
        observation = observation.reshape((-1, *obs_shape))
        if gail:
            rew = [
                reward_giver.get_reward(obs, act)
                for obs, act in zip(observation, clipped_action)
            ]
        else:
            rew = true_rew
        data['rew'].extend(rew)
        data['true_rew'].extend(true_rew)
        data['dones'].extend(done)

        current['ep_rets'] += np.mean(rew)
        current['ep_true_rets'] += np.mean(true_rew)
        current['ep_lens'] += 1
        current_it_len += 1
        if np.any(done):
            for key in current:
                segments[key].append(current[key])
            current.clear()
            if not isinstance(env, VecEnv):
                observation = env.reset().reshape((-1, *obs_shape))


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory
    (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(seg["dones"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(rew_len, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - new[step + 1]
        delta = rew[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        gaelam[step] = lastgaelam = delta + \
            gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return list(chain.from_iterable(listoflists))
