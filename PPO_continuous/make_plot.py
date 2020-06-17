import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='DQN Training for Hitman GO')
parser.add_argument('--end_iter', type=int, default=1000,
                    help='End Iteration')
args = parser.parse_args()

env = gym.make('CartPole-v1')
a_size=env.action_space.n
chk_dir='./chk_cartpole/'

avg_records=[]
# for chk in checkpoints:
for iter in range(10,args.end_iter+10,10):
    net=keras.models.load_model('./chk_cartpole/actor'+str(iter)+'.h5')
    #avg 100 rewards
    avg_100=0
    for i in range(100):
        s=env.reset()
        while True:
            pi=net.predict(np.expand_dims(s,axis=0))
            prob=pi[0]
            prob /= prob.sum()
            a = np.random.choice(range(a_size), p=prob)

            #action,value =net.get_action(np.expand_dims(s,axis=0))

            next_s, reward, done, _ = env.step(a)
            
            reward_sum += reward

            if done:
                break
            steps+=1

            s = next_s
        avg_100+=reward_sum
    avg_records.append(avg_100/100)

plt.plot(np.arange((len(avg_records))), avg_records, label='PPO Cartpole')
fig =plt.gcf()
plt.savefig("cartpole_ppo.png")
plt.show()