import tensorflow as tf
import time
import csv
from environment import Env
from actor import ActorNetwork         #策略神经网络
from lagrange import LagrangeNetwork   #值神经网络
from experience_replay import ExpReplay, Step
from agent import Agent
import os
import pickle
import numpy as np
import scipy.io as scio
import psutil
import gc
from memory_profiler import profile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1: use CPU for training, 0: GPU
np.seterr(all='raise')

def save_model(global_vars, fname):
    f = open(fname, 'wb')
    pickle.dump(global_vars, f)
    f.close()

def load_model(fname):
    f = open(fname, 'rb')
    var = pickle.load(f)
    f.close()
    return var

if __name__ == '__main__':
    data = scio.loadmat('./train_330-330e-550-770_nRB_0829.mat')
    data = np.array(data['sample'])
    num_monte_carlo = 1
    num_episodes = 200010          #300010
    pre_train_episode = 10000
    anneal_episode = 10000
    train_episode = 180000
    rate_ini = 0
    rate_end = 0
    eval_every = 100

    critic_lr = 0.001
    actor_lr = 0.001

    batch_size = 32
    start_mem = 128
    mem_size = 100000

    actor_num_hidden_1 = 50    #策略神经网络
    actor_num_hidden_2 = 40
    actor_num_hidden_3 = 30

    lag_num_hidden_1 = 200     #值神经网络
    lag_num_hidden_2 = 150

    env = Env()

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    actor = ActorNetwork(actor_lr, actor_num_hidden_1, actor_num_hidden_2, actor_num_hidden_3)
    lag = LagrangeNetwork(critic_lr, lag_num_hidden_1, lag_num_hidden_2)

    start_time = time.clock()

    for k in range(num_monte_carlo):
        exp_rep = ExpReplay(mem_size=mem_size, start_mem=start_mem, batch_size=batch_size)
        agent = Agent(actor, lag,  exp_rep)

        state_list = []
        action_list = []
        reward_list = []

        cum_reward = 0
        cum_loss = 0
        average_reward_list = []
        average_loss_list = []

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            noise_rate = rate_ini
            cur_state = env.reset(data)

            for i in range(num_episodes):

                if pre_train_episode <= i < pre_train_episode + anneal_episode:
                    noise_rate = noise_rate - (rate_ini - rate_end) / anneal_episode
                elif pre_train_episode + anneal_episode <= i < pre_train_episode + anneal_episode + train_episode:
                    noise_rate = rate_end
                elif pre_train_episode + anneal_episode + train_episode <= i:
                    noise_rate = 0

                if (i + 1) % eval_every == 0:
                    action = agent.get_action(cur_state, sess)
                else:
                    action = agent.get_action_noise(cur_state, sess, noise_rate)

                next_state, reward = env.step(action,cur_state,data)
                agent.add_step_to_exp(Step(cur_state, action, reward))

                cum_reward = cum_reward + reward

                if i <= pre_train_episode + anneal_episode + train_episode:
                    loss = agent.learn_batch(sess)
                else:
                    loss = 0


                cum_loss = cum_loss + loss

                if (i + 1) % eval_every == 0:
                    end_time = time.clock()

                    average_reward_list.append(cum_reward/eval_every)
                    average_loss_list.append(cum_loss/eval_every)

                    print("Monte Carlo {}: Episode {}-{}, average_loss: {} "
                          " average_reward: {}, test_reward: {}\n"
                          "state:{}\n"
                          "action{}".
                          format(k, i - eval_every + 1, i, cum_loss/eval_every,
                                  cum_reward/eval_every,reward, cur_state, action))
                    print("running time: {}".format(end_time - start_time))

                    start_time = time.clock()
                    cum_reward = 0
                    cum_loss = 0

                    with open('./results/0829/average_reward_train_200010_' + str(k) + '.csv', 'w', newline='') as logfile:
                        wr = csv.writer(logfile)
                        wr.writerow(average_reward_list)
                    with open('./results/0829/average_loss_train_200010_' + str(k) + '.csv', 'w', newline='') as logfile:
                        wr = csv.writer(logfile)
                        wr.writerow(average_loss_list)

                if i == pre_train_episode + anneal_episode + train_episode:
                    actor_var = actor.get_actor_variable(sess)
                    critic_var = lag.get_critic_variable(sess)
                    save_model(actor_var,
                               './results/0829/actor_0825_200010_{}'.format(k))
                    save_model(critic_var,
                               './results/0829/critic_0825_200010_{}'.format(k))

                print('{}：%.2f MB'.format(i) % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
                # testing
                if i > pre_train_episode + anneal_episode + train_episode:
                    state_list.append(cur_state)
                    action_list.append(action)
                    reward_list.append(reward)

                    with open('./results/0829/Test_reward_train_200010_' + str(k) + '.csv', 'w', newline='') as logfile:
                        wr = csv.writer(logfile)
                        wr.writerow(reward_list)
                    with open('./results/0829/Test_state_train_200010_' + str(k) + '.csv', 'w', newline='') as logfile:
                        wr = csv.writer(logfile)
                        wr.writerow(state_list)
                    with open('./results/0829/Test_action_train_200010_' + str(k) + '.csv', 'w', newline='') as logfile:
                        wr = csv.writer(logfile)
                        wr.writerow(action_list)

                cur_state = next_state

