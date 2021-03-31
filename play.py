#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np
import gnwrapper

import torch

from lib import wrappers
from lib import dqn_model

import collections

FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=torch.cuda.is_available(),
                        action="store_true", help="Enable cuda")
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-w", "--wrapper", required=True, choices=range(1, 7),
                        type=int, help="Wrapper to use")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    wr, env = wrappers.get_env(args.wrapper)
    env = gnwrapper.Monitor(env, directory="./", force=True)

    # fix action space to 4 to make compatible with breakout
    action_space = 4

    net = dqn_model.DQN(env.observation_space.shape, action_space)
    checkpoint = torch.load(
        args.model, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)

    env.env.close()
