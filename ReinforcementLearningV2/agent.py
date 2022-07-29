import numpy as np

class Agent:
    def __init__(self, actor, lag, exp_rep):
        self.actor = actor
        self.lag = lag
        self.exp_rep = exp_rep

    def add_step_to_exp(self, step):
        self.exp_rep.add_step(step)

    def get_action(self, state, sess):
        action = self.actor.get_action([state], sess)[0]
        return action

    def get_action_noise(self, state, sess, noise_rate):
        action = self.actor.get_action([state], sess)[0]
        action = max(0, action + np.random.rand()*noise_rate)

        return action

    def learn_batch(self, sess):
        batch = self.exp_rep.sample()

        if len(batch) == 0:
            return 0

        state =[step.state for step in batch]
        action = [step.action for step in batch]
        reward = [step.reward for step in batch]

        loss = self.lag.train(state, action, reward, sess)

        cur_action = self.actor.get_action(state, sess)

        lag_grad_action = self.lag.get_grad(state, cur_action, sess)

        # update actor policy with sampled gradient
        self.actor.train(state, lag_grad_action[0][0], sess)

        return loss
