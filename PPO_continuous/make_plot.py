import matplotlib.pyplot as plt
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np

env = gym.make('CartPole-v1')
chk_dir='./chk_cartpole/

avg_records=[]
# for chk in checkpoints:
for iter in range(10,1010,10):
    net=keras.models.load_model('./chk_cartpole/actor'+str(iter)+'.h5')
    #avg 100 rewards
    avg_100=0
    for i in range(100):
        s=env.reset()
        while True:
            action,value = self.net.get_action(np.expand_dims(s,axis=0))

            next_s, reward, done, _ = self.env.step(action)
            
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