###################################调用保存的模型，测试################################
import tensorflow as tf
from actor import ActorNetwork
import pickle
import numpy as np
import time
import os
import psutil
import gc
from memory_profiler import profile


def load_model(fname):
    f = open(fname, 'rb')
    var = pickle.load(f)
    f.close()
    return var

def test(state):
    tf.compat.v1.reset_default_graph()

    actor_lr = 0.001

    actor_num_hidden_1 = 50
    actor_num_hidden_2 = 40
    actor_num_hidden_3 = 30

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    actor = ActorNetwork(actor_lr, actor_num_hidden_1, actor_num_hidden_2, actor_num_hidden_3)

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        global_vars = load_model('results/0905/actor_0905_200010_liji_duobu_0')
        actor.set_actor_variable(sess, global_vars)
        start_time = time.time()
        pre1 = actor.get_action([state], sess)[0][0]
        end_time = time.time()
        print('Consumed Time:{:.5f}s\n'.format(end_time - start_time))
        # pre1 = round(pre1)
        print(pre1)

        return pre1

@profile
def test_memory():
    a = np.full(shape=(600, 700), fill_value=99.0)
    return a

if __name__ == '__main__':
    print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    test([0.33, 1.2, 0.4, 0.3])
    print('B：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))