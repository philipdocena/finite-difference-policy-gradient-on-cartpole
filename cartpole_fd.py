# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:24:56 2018

@author: Philip Docena
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gym


def policy_pi_w(s,w):
    """ select action via softmax probability """
    a_probs=np.dot(s,w)
    a=np.argmax(np.exp(a_probs)/sum(np.exp(a_probs)))
    return a


def expected_return_sigma(gamma,r_ep):
    """ discount rewards via dynamic programming to avoid the large exponents """
    gamma_disc=np.zeros(len(r_ep))
    r_disc=r_ep[0]
    gamma_disc[0]=1
    for i in range(1,len(r_ep)):
        gamma_disc[i]=gamma*gamma_disc[i-1]
        r_disc+=gamma_disc[i]*r_ep[i]
    return r_disc


def regression(dw,dJ,w_shape):
    """ closed form regression to calculate gradient """
    w_pseudo=np.dot(np.linalg.pinv(np.dot(dw.T,dw)),dw.T)
    w=np.dot(w_pseudo,dJ)
    w=w.reshape(w_shape)
    return w


def main():
    cartpole_type='CartPole-v0'
    #cartpole_type='CartPole-v1'
    env=gym.make(cartpole_type)
    target_reward=195 if cartpole_type=='CartPole-v0' else 475

    env_seed,numpy_seed,env_testing_seed=0,0,123
    env.seed(env_seed)
    np.random.seed(numpy_seed)

    num_sensors,num_actions=4,env.action_space.n
    weight_random_init,rhc_epsilon=1e-4,1e-2
    max_train_iters,min_rollouts,test_iters=500,100,200
    alpha,gamma=0.001,0.99
    fd_batch=1

    # model a linear combo parameter vector per action, i.e., a fully connected 4-node hidden layer
    w=np.random.rand(num_sensors,num_actions)*weight_random_init

    # training phase
    print('training to find an optimal policy')
    r_training=[]
    num_rollouts=0

    while True:
        w_save=np.copy(w)        
        
        dw,dJ=[],[]
        for fd_reg_i in range(fd_batch):
        
            # finite difference via central difference: J(w+dw)-J(w-dw)
            dw_curr=np.random.rand(w.shape[0],w.shape[1])*rhc_epsilon
            
            w_pair,J_pair=[],[]
            for fd_i in range(2):
                w=w+dw_curr if fd_i%2==0 else w_save-dw_curr
                
                s1=env.reset()
                r_ep=[]
                
                while True:
                    #env.render()
                    a=policy_pi_w(s1,w)
                    s2,r,done,_=env.step(a)
                    r_ep.append(r)
                    
                    if done: break
                    s1=s2
                
                r_training.append(np.sum(r_ep))
                J=expected_return_sigma(gamma,r_ep)
                
                J_pair.append(J)
                w_pair.append(w)
            
            dw.append((w_pair[0]-w_pair[1]).ravel())
            dJ.append(J_pair[0]-J_pair[1])
            
        # regression to find gradient; very noisy due to different reference w and trajectories!
        grad_w=regression(np.array(dw),np.array(dJ),w.shape)
        
        w=w_save+alpha*(1-num_rollouts/max_train_iters)*rhc_epsilon*grad_w
        
        num_rollouts+=2*fd_batch
        if num_rollouts>max_train_iters:
            print('RHC reached maximum iterations of {}.'.format(max_train_iters))
            break
        if np.mean(r_training[-100:])>target_reward and num_rollouts>=min_rollouts:
            print('RHC reached the target reward of {} on episode {}.'.format(target_reward,num_rollouts))
            break
            
    # testing phase
    print('testing final policy')
    env.seed(env_testing_seed)
    r_testing=[]
    #for i in range(test_iters):
    for i in range(max(num_rollouts,100)):
        s=env.reset()
        r_ep=0
        
        while True:
            #env.render()
            a=policy_pi_w(s,w)
            s,r,done,_=env.step(a)
            r_ep+=r
            
            if done: break
        
        #print('episode: {}  reward: {:.2f}'.format(i,r_ep))
        r_testing.append(r_ep)
        
    print('average reward on testing',np.mean(r_testing))
    env.close()

    # plot results
    plt.plot(r_training,label='training, moving ave (100)')
    plt.plot(r_testing,label='testing, per episode')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Finite Difference Policy Optimization on {}\n \
env seed={}, training seed={}, testing seed={} \n \
Philip Docena'.format(cartpole_type,env_seed,numpy_seed,env_testing_seed),fontsize=10)
    plt.legend(loc='best')
    plt.show()


if __name__=='__main__':
    main()