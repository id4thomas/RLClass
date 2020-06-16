#PPO for Cartpole-v0
import numpy as np
import tensorflow as tf
from tensorflow import keras

import gym
import pybullet_envs

import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import os
import random

from tensorflow.keras import layers
#my utilss
from utils import Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ACModel():
    def __init__(self,env):
        self.a_size=env.action_space.n
        self.s_size=env.observation_space.shape[0]

        #self.action_low=env.action_space.low
        #self.action_high=env.action_space.high

        self.actor_lr=1e-4
        self.critic_lr=1e-3
        self.lr=1e-4
        #self.actor,self.critic=self.make_model()
        self.actor=self.make_actor()
        self.critic=self.make_critic()

        self.gamma=0.99

        #PPO Parameters
        self.ppo_epochs=10
        self.batch_size=64
        self.cliprange=0.2

        #self.c_op=tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        #self.a_op=tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        #self.c_op=tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.op=tf.keras.optimizers.Adam(learning_rate=self.lr)

    def make_critic(self):
        in1=keras.layers.Input(shape=(self.s_size,))
        d1=tf.keras.layers.Dense(64, activation='tanh',kernel_initializer='he_uniform')(in1)
        d2=keras.layers.Dense(64, activation='tanh',kernel_initializer='he_uniform')(d1)
        #critic
        s_v=tf.keras.layers.Dense(1,kernel_initializer='he_uniform',name="s_v")(d2)
        critic=keras.models.Model(inputs=in1,outputs=s_v)
        #critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.critic_lr),loss='mean_squared_error')
        return critic

    def make_actor(self):
        in1=keras.layers.Input(shape=(self.s_size,))
        d1=tf.keras.layers.Dense(64, activation='tanh',kernel_initializer='he_uniform')(in1)
        d2=keras.layers.Dense(64, activation='tanh',kernel_initializer='he_uniform')(d1)
        pi=tf.keras.layers.Dense(self.a_size, activation='softmax',kernel_initializer='he_uniform',name="pi")(d2)
        #sigma=tf.keras.layers.Dense(self.a_size,activation=tf.keras.activations.exponential,kernel_initializer='he_uniform',name="sigma")(d2)
        
        actor=keras.models.Model(inputs=in1,outputs=pi)
        return actor
    
    def get_action(self,state):
        pi=self.actor(state,training=False)
        #sigma=self.sigma
        #print(pi,pi.shape)
        val=self.critic(state,training=False)
        pi[0] /= pi[0].sum()
        a = np.random.choice(range(self.a_size), p=pi[0])
        #print(a)
        #print(mu,sigma)
        #random sample action
        #a=np.random.normal(loc=mu, scale=sigma)[0]
        #a=mu + tf.exp(sigma) * tf.random.normal(tf.shape(mu))
        #print(a)
        return a,val

    '''
    def get_logp(self,s,a):
        mu=self.actor(s,training=True)
        sigma=self.sigma

        e1=0.5 * tf.reduce_sum(tf.square((a - mu) / (tf.exp(sigma))), axis=-1) 
        e2=0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(mu)[-1],tf.float32)
        e3=tf.reduce_sum(sigma, axis=-1)
        
        logp = -(e1+e2+e3 )
        return logp

    def get_logp2(self,s,a):
        mu=self.actor(s,training=True)
        sigma=self.sigma
        normal_dist=tfp.distributions.Normal(loc=mu,scale=sigma)
        log_p = normal_dist.log_prob(a)
        return log_p'''


    def apply_grads(self,a_grads,c_grads):
    #def apply_grads(self,grads,trainables):
        grads=a_grads+c_grads
        trainables=self.actor.trainable_weights+self.critic.trainable_weights
        #print(trainables)
        self.op.apply_gradients(zip(grads,trainables))
        #self.op.apply_gradients(zip(grads,trainables))
        #self.a_op.apply_gradients(zip(a_grads,self.actor.trainable_weights))
        #print(self.actor.trainable_weights)
        #print(a_grads)
        #self.a_op.apply_gradients(zip([sig_grad],[self.sigma]))
        #self.c_op.apply_gradients(zip(c_grads,self.critic.trainable_weights))



class PPO():
    def __init__(self,env,num_iters):
        self.env=env
        
        #models
        self.net=ACModel(env)
        self.old_net=ACModel(env)

        #PPO parameters
        self.num_iters=num_iters
        self.num_epochs=10
        self.batch_size=64
        self.clip_range=0.2

        #For entropy for exploration
        self.ent_coef=0.001

        self.utils=Utils()
        #self.trainables=self.net.actor.trainable_weights+[self.net.sigma]+self.net.critic.trainable_weights

    def save_models(self,iter):
        #save model
        self.net.critic.save('./chk_ppo/critic'+str(iter)+'.h5')
        self.net.actor.save('./chk_ppo/actor'+str(iter)+'.h5')

    def get_samples(self):
        #Run episode
        s = self.env.reset()
        reward_sum = 0
        records=[]
        steps=0
        while True:
            #env.render()
            action,value = self.net.get_action(np.expand_dims(s,axis=0))

            next_s, reward, done, _ = self.env.step(action)
            
            reward_sum += reward

            record=(s,action,reward,next_s,done,value)
            records.append(record)

            if done:
                break
            steps+=1

            s = next_s
        return records,reward_sum,steps

    def calc_grads(self,batch):
        s=np.array([mem[0] for mem in batch])#batch state
        a=np.array([mem[1] for mem in batch])#batch action
        td=np.array([mem[2] for mem in batch])#batch td
        adv=np.array([mem[3] for mem in batch])

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.net.actor.trainable_weights)
            t.watch(self.net.critic.trainable_weights)
            s = np.transpose(s, (0, 2, 3, 1))
            net_pi=self.net.actor(s)
            old_net_pi=self.old_net.actor(s)

            row_indices = tf.range(len(net_pi))
            a_indices = tf.stack([row_indices, a], axis=1)

            # retrieve values by indices
            pi = tf.gather_nd(net_pi, a_indices)
            old_pi = tf.gather_nd(old_net_pi, a_indices)

            ratio=tf.exp(tf.math.log(pi)-tf.stop_gradient(tf.math.log(old_pi)))
            pg_loss1=adv*ratio
            #print(pg_loss1[0])
            pg_loss2=adv*tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            #batch grad
            a_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
            v_pred=self.net.critic(s,training=True)
            c_loss = tf.reduce_mean(tf.square(v_pred - td))

            loss=a_loss+c_loss
        
        a_grads=t.gradient(loss,self.net.actor.trainable_weights)
            
        c_grads=t.gradient(loss,self.net.critic.trainable_weights)
        #grads=t.gradient(loss,self.trainables)
        return a_grads,c_grads,a_loss,c_loss
        #return grads,a_loss,c_loss

    def run(self):
        avg = 0
        avg_iter=10

        #PPO Iteration
        
        cur_iter=0

        #Trajectory collection
        num_steps=3072
        #num_steps=10
        #avg_100=0
        
        while cur_iter<self.num_iters:
            T=[]
            score_sum=0
            print("\n\nIteration",cur_iter)
            #update old net
            #print(self.old_net.sigma)
            self.old_net.actor.set_weights(self.net.actor.get_weights())
            self.old_net.critic.set_weights(self.net.critic.get_weights())
            #print(self.old_net.sigma)
            #get trajectories
            cur_steps=0
            cur_episode=0 #collected episodes
            while cur_steps<num_steps:
                trajectory,score,steps=self.get_samples()
                score_sum+=score
                cur_steps+=steps
                T.append(trajectory)
                cur_episode+=1
                print('Reward {} Steps {}'.format(score,steps))

            #Train
            self.train(T)
            cur_iter+=1
            avg += score_sum/cur_episode

            print(f"run {cur_iter} total reward: {score_sum/cur_episode}")
            #save every avg_iter iterations
            if (cur_iter)%avg_iter==0:
                print(f"average {cur_iter} total reward: {avg/avg_iter}")
                avg=0
                self.save_models(cur_iter)


    def train(self,T):
        #Get values
        vals=self.utils.calc_traj_vals(T)


        for ep in range(self.num_epochs):
            #epochs sgd
            #make batch
            batch_iters=int(len(vals)/self.batch_size)
            #print('batch',batch_iters,len(vals))
            v_loss=0
            p_loss=0
            if batch_iters==0:
                a_grads,c_grads,p_loss,v_loss=self.calc_grads(vals)
                #grads,p_loss,v_loss=self.calc_grads(vals)
                self.net.apply_grads(a_grads,c_grads)
                #self.net.apply_grads(grads,self.trainables)
            else:
                for i in range(batch_iters):
                    #run batch
                    cur_idx=i*self.batch_size
                    batch=vals[cur_idx:cur_idx+self.batch_size]

                    a_grads,c_grads,a_loss,c_loss=self.calc_grads(batch)
                    #grads,a_loss,c_loss=self.calc_grads(vals)
                    self.net.apply_grads(a_grads,c_grads)
                    #self.net.apply_grads(grads,self.trainables)
                    v_loss+=c_loss
                    p_loss+=a_loss
                v_loss/=batch_iters
                p_loss/=batch_iters

            print("vf_loss: {:.5f}, pol_loss: {:.5f}".format(v_loss, p_loss))
            



def main():
    #env = gym.make('AntBulletEnv-v0')
    env= gym.make('CartPole-v0')
    num_iters=1000000
    #tf.debugging.set_log_device_placement(True)
    ppo=PPO(env,num_iters)

    ppo.run()

if __name__ == "__main__":
    main()
