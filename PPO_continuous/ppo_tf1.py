import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
import argparse

import gym
import pybullet_envs

import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import os
import random

from utils import Utils

tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#A2C Model

class ACModel():
    def __init__(self,env,name):
        self.scope = name
        self.a_size=env.action_space.shape[0]
        self.s_size=28

        self.action_low=env.action_space.low
        self.action_high=env.action_space.high

        
        self.s=tf.placeholder(shape=[None, self.s_size],
                                        name="s", dtype=tf.float32)

        self.a=tf.placeholder(shape=[None, self.a_size],
                                        name="a", dtype=tf.float32)                        

        self.adv = tf.placeholder(shape=[None],name="adv", dtype=tf.float32)
        self.td = tf.placeholder(shape=[None],name="td", dtype=tf.float32)

        #self.actor,self.critic=self.make_model()
        self.trainable=True
        self.mu,self.sigma,self.v=self.make_model()

        self.get_a=self.mu + tf.exp(self.sigma) * tf.random_normal(tf.shape(self.mu))

    def make_model(self):
        with tf.variable_scope(self.scope):
            #actor
            in1=self.s
            d1=tf.layers.dense(in1, units=128, activation=tf.nn.tanh, name="a_d1",trainable=self.trainable)
            d2=tf.layers.dense(d1, units=128, activation=tf.nn.tanh, name="a_d2",trainable=self.trainable)
            mu_out=tf.layers.dense(d2, units=self.a_size, activation=tf.nn.tanh, name="mu_out",trainable=self.trainable)
            sigma_out = tf.get_variable(name="", shape=[self.a_size],
                                    initializer=tf.zeros_initializer)


            #critic
            in1=self.s
            d1=tf.layers.dense(in1, units=128, activation=tf.nn.tanh, name="c_d1",trainable=self.trainable)
            d2=tf.layers.dense(d1, units=128, activation=tf.nn.tanh, name="c_d2",trainable=self.trainable)

            val_out=tf.layers.dense(d2, units=1, activation=None, name="val_out",trainable=self.trainable)
            #print(mu_out,sigma_out)
        return mu_out,sigma_out,val_out

    def get_action(self,state):
        
        a,value=tf.get_default_session().run([self.get_a,self.v],feed_dict={self.s: state})

        return a[0],value[0][0]


    def get_logp(self):
        e1=0.5 * tf.reduce_sum(tf.square((self.a - self.mu) / (tf.exp(self.sigma))), axis=-1) 
        e2=0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.mu)[-1])
        e3=tf.reduce_sum(self.sigma, axis=-1)
        
        logp = -(e1+e2+e3 )
        return logp


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


class PPO():
    def __init__(self,env):
        #Make Session
        sess = tf.InteractiveSession()
        
        self.env=env
        self.lr = 1e-4
        #self.critic_lr=0.001

        self.gamma=0.99
        self.gae=0.95
        self.net=ACModel(env,'net',)
        self.old_net=ACModel(env,'old')

        #PPO Parameters
        self.ppo_epochs=10
        self.batch_size=64
        self.clip_range=0.2

        #For entropy for exploration
        self.ent_coef=0.01
        
        self.saver = tf.train.Saver(max_to_keep=5000)
        self.build_update()
        self.build_update_models()

        self.utils=Utils()
        tf.get_default_session().run(tf.global_variables_initializer())

        
    def build_update(self):
        ratio=tf.exp(self.net.get_logp() - tf.stop_gradient(self.old_net.get_logp()))

        surr1 = ratio * self.net.adv
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)*self.net.adv

        self.a_loss=-tf.reduce_mean(tf.minimum(surr1,surr2))
        self.v_loss=tf.reduce_mean(tf.square(self.net.v - self.net.td))

        self.total_loss=self.a_loss + 10*self.v_loss
        self.op=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    def build_update_models(self):
        assign_op = []
        for (newv, oldv) in zip(self.net.get_variables(), self.old_net.get_variables()):
            assign_op.append(tf.assign(oldv, newv))
        self.assign_op=assign_op

    def train(self,T):
        vals=self.utils.calc_traj_vals
        for ep in range(self.ppo_epochs):
            batch_iters=int(len(vals)/self.batch_size)
            #print('batch',batch_iters,len(vals))
            val_loss=0
            pol_loss=0
            if batch_iters==0:
                s=np.array([mem[0] for mem in vals],dtype='float32')#batch state
                a=np.array([mem[1] for mem in vals],dtype='float32')#batch action
                td=np.array([mem[4] for mem in vals],dtype='float32')#batch td
                adv=np.array([mem[3] for mem in vals],dtype='float32')

                v_loss,a_loss,_=tf.get_default_session().run([self.v_loss,self.a_loss,self.op],feed_dict={self.net.s: s,
                self.net.a: a,
                self.net.adv:adv,
                self.net.td:td,
                self.old_net.s: s,
                self.old_net.a: a,
                self.old_net.adv:adv,
                self.old_net.td:td
                })
                val_loss+=v_loss
                pol_loss+=a_loss
                batch_iters=1
                #print("vf_loss: {:.5f}, pol_loss: {:.5f}".format(val_loss, pol_loss))
            else:
                for i in range(batch_iters):
                    cur_idx=i*self.batch_size
                    batch=vals[cur_idx:cur_idx+self.batch_size]

                    s=np.array([mem[0] for mem in batch],dtype='float32')#batch state
                    a=np.array([mem[1] for mem in batch],dtype='float32')#batch action
                    td=np.array([mem[4] for mem in batch],dtype='float32')#batch td
                    adv=np.array([mem[3] for mem in batch],dtype='float32')
                    v_loss,a_loss,_=tf.get_default_session().run([self.v_loss,self.a_loss,self.op],feed_dict={self.net.s: s,
                    self.net.a: a,
                    self.net.adv:adv,
                    self.net.td:td,
                    self.old_net.s: s,
                    self.old_net.a: a,
                    self.old_net.adv:adv,
                    self.old_net.td:td
                    })
                    val_loss+=v_loss
                    pol_loss+=a_loss
        print("vf_loss: {:.5f}, pol_loss: {:.5f}".format(val_loss/batch_iters, pol_loss/batch_iters))

                

    def run(self):
        avg = 0
        num_iters=100000
        avg_100=0
        cur_episode=0
        cur_iter=0
        sample_ep=3
        num_steps=3072
        while cur_iter<num_iters:
            #Get Trajectories
            T=[]
            score_sum=0
            print("Iteration",cur_iter)
            tf.get_default_session().run(self.assign_op)
            cur_steps=0
            cur_episode=0
            while cur_steps<num_steps:
                trajectory,score,steps=self.get_samples()
                score_sum+=score
                cur_steps+=steps
                T.append(trajectory)
                cur_episode+=1
                print('Reward {}'.format(score))
            print(f"T average {cur_iter} total reward: {score_sum/cur_episode}")
            #Train
            self.train(T)
            cur_iter+=1
            avg += score_sum/cur_episode
            avg_100+=score_sum/cur_episode
            print('\n\n')

            if (cur_iter)%10==0:
                print(f"average {cur_iter} total reward: {avg_100/100}")
                avg_100=0
                #self.save_models(cur_iter)

    def save_models(self,episode):
        self.saver.save(tf.get_default_session(), './chk_final/',global_step=episode)

    def get_samples(self):
        s = self.env.reset()
        reward_sum = 0
        records=[]
        steps=0
        
        while True:
            #env.render()
            action,value = self.net.get_action(np.expand_dims(s,axis=0))

            next_s, reward, done, _ = env.step(action)
            
            reward_sum += reward

            record=(s,action,reward,next_s,done,value)
            records.append(record)

            if done:
                break
            steps+=1

            s = next_s
        return records,reward_sum,steps



if __name__ == "__main__":

    env = gym.make('AntBulletEnv-v0')
    sess = tf.InteractiveSession()
    ppo=PPO(env)
    tf.get_default_session().run(tf.global_variables_initializer())
    ppo.run()
