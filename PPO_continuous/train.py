#import tensorflow as tf
import argparse
import importlib

#Import Environment
import gym
import pybullet_envs


from ppo_tf1 import PPO as ppo_tf1
from ppo_tf2 import PPO as ppo_tf2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Params')
    parser.add_argument('--mode', default='tf2',
                        help='TF1 or TF2')
    args = parser.parse_args()

    if args.mode=='tf2':
        mod=importlib.import_module('ppo_tf2')
    else:
        mod=importlib.import_module('ppo_tf1')

    env = gym.make('AntBulletEnv-v0') 
    ppo=getattr(mod, 'PPO')(env)
    ppo.run()