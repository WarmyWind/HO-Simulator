import tensorflow as tf
#值神经网络

class LagrangeNetwork:

    def __init__(self, learning_rate, num_hidden_1=400, num_hidden_2=300):

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.regularization = 0
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.weights_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN")

        self.input_state, self.action, self.q_value, self.critic_variables =\
            self._build_network("lagrange")

        self.target = tf.placeholder(tf.float32, [None, 1])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.critic_variables[0:6:2]])
        self.loss = tf.reduce_mean(tf.square(self.target - self.q_value)) + self.regularization * self.l2_loss
        self.optimize = self.optimizer.minimize(self.loss)

        self.lag_grad_action = tf.gradients(self.q_value, self.action)

    def _build_network(self, name):
        input_state = tf.placeholder(tf.float32, [None, 3], name="input_state")
        action = tf.placeholder(tf.float32, [None, 1], name="action")

        with tf.variable_scope(name):
            layer_1 = tf.contrib.layers.fully_connected(tf.concat([input_state, action], 1),
                                                        self.num_hidden_1, activation_fn=tf.nn.relu,weights_initializer=self.weights_initializer)
            layer_2 = tf.contrib.layers.fully_connected(layer_1, self.num_hidden_2, activation_fn=tf.nn.relu,
                                                        weights_initializer=self.weights_initializer)

            q_value = tf.contrib.layers.fully_connected(layer_2, 1, activation_fn=None,
                                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))


        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_state, action, q_value, critic_variables

    def get_grad(self, state, action, sess):
        grad_action = sess.run([self.lag_grad_action], feed_dict={
            self.input_state: state,
            self.action: action
        })
        return grad_action

    def train(self, state, action, target, sess):
        _, loss = sess.run([self.optimize, self.loss], feed_dict={
            self.input_state: state,
            self.action: action,
            self.target: target
        })
        return loss

    def get_critic_variable(self, sess):
        return sess.run(self.critic_variables)

    def set_critic_variable(self, sess, var):
        critic_var = self.critic_variables
        for variable, value in zip(critic_var, var):
                variable.load(value, sess)