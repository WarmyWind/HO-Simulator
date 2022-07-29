import tensorflow as tf
import tf_slim as slim
#策略神经网络

class ActorNetwork:
    def __init__(self, learning_rate, num_hidden_1, num_hidden_2, num_hidden_3):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.num_hidden_3 = num_hidden_3

        self.weights_initializer = slim.variance_scaling_initializer(mode="FAN_IN")

        self.input_state, self.action, self.actor_variables = self._build_network("actor")

        self.critic_gradients_action = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.actor_gradients = tf.gradients(self.action, self.actor_variables, -self.critic_gradients_action)

        # the minus above means we are maximizing the expected return
        self.train_op = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_variables))

    def _build_network(self, name):
        tf.compat.v1.disable_eager_execution()
        input_state = tf.compat.v1.placeholder(tf.float32, [None, 3])
        with tf.compat.v1.variable_scope(name):
            layer_1 = slim.layers.fully_connected(input_state, self.num_hidden_1,
                                                        weights_initializer=self.weights_initializer)
            layer_2 = slim.layers.fully_connected(layer_1, self.num_hidden_2,
                                                        weights_initializer=self.weights_initializer)
            layer_3 = slim.layers.fully_connected(layer_2, self.num_hidden_3,
                                                        weights_initializer=self.weights_initializer)
            action = slim.layers.fully_connected(layer_3, 1, activation_fn=tf.nn.relu,
                                                       weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                       biases_initializer=tf.constant_initializer(10))  # 10

        actor_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_state, action, actor_variables

    def get_action(self, state, sess):
        return sess.run(self.action, feed_dict={self.input_state: state})

    def train(self, state, critic_gradients_action, sess):
        batch_size = len(state)
        return sess.run(self.train_op, feed_dict={self.input_state: state,
                                                  self.critic_gradients_action: critic_gradients_action / batch_size})

    def get_actor_variable(self, sess):
        return sess.run(self.actor_variables)

    def set_actor_variable(self, sess, var):
        actor_var = self.actor_variables
        for variable, value in zip(actor_var, var):
                variable.load(value, sess)