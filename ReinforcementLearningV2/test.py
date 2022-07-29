###################################调用保存的模型，测试################################
import tensorflow as tf
from actor import ActorNetwork
import pickle

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
        global_vars = load_model('actor_1234567(4-30_zscore)_0713_12010_0')
        actor.set_actor_variable(sess, global_vars)
        pre1 = actor.get_action([state], sess)[0][0]
        pre1 = round(pre1)
        print(pre1)

        return pre1


if __name__ == '__main__':
    test([1.06666667, 2, 0.3])
