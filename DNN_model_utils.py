import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import sys


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma)

    return - (log_coeff + exponent).sum()


class BaseNet(object):
    def __init__(self):
        cprint('c', '\nNet:')

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.learn_rate *= gamma
                print('learning rate: %f  (%d)\n' % self.learn_rate, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learn_rate

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({
            # 'epoch': self.epoch,
            # 'lr': self.learn_rate,
            'model': self.network,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        state_dict = torch.load(filename)
        # self.epoch = state_dict['epoch']
        # self.learn_rate = state_dict['lr']
        self.network = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring model, lr: %f' % (self.learn_rate))
        return


class DNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units):
        super(DNN_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = nn.Linear(self.input_dim, no_units, bias=True)
        self.layer2 = nn.Linear(no_units, no_units, bias=True)
        # self.layer3 = nn.Linear(no_units, no_units, bias=True)
        self.layer3 = nn.Linear(no_units, self.output_dim, bias=True)

        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.activation(x)

        # x = self.layer3(x)
        # x = self.activation(x)

        x = self.layer3(x)

        return x


class DNN_Model_Wrapper(BaseNet):
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.no_units = no_units
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        # self.no_batches = no_batches
        # self.schedule = None  # [] #[50,200,400,600]
        self.network = DNN_Model(input_dim=input_dim, output_dim=output_dim,
                                               no_units=no_units)
        self.network.cuda()
        self.epoch = 0
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learn_rate)
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learn_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=6, factor=0.35, verbose=True, min_lr=1e-5)

        self.criteria = nn.MSELoss()

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)

        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0

        # for i in range(no_samples):
        output = self.network(x)

        # calculate fit loss based on mean and standard deviation of output
        total_loss = self.criteria(output, y.view(-1, self.output_dim))
        # print("fit_loss_total:",fit_loss_total)

        total_loss.backward()
        self.optimizer.step()
        # self.scheduler.step(total_loss)

        return total_loss * len(y)

    def test(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)

        # reset gradient and total loss
        self.optimizer.zero_grad()
        fit_loss_total = 0

        # for i in range(no_samples):
        output = self.network(x)

        # calculate fit loss based on mean and standard deviation of output
        total_loss = self.criteria(output, y.view(-1, self.output_dim))
        return total_loss * len(y)

    def set_train(self):
        self.network.train()
        return

class DNN_Hetero_MCDropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, p=0.5):
        super(DNN_Hetero_MCDropout_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # network with two hidden and one output layer
        self.layer1 = nn.Linear(self.input_dim, no_units, bias=True)
        self.layer2 = nn.Linear(no_units, no_units, bias=True)
        # self.layer3 = nn.Linear(no_units, no_units, bias=True)
        self.layer3 = nn.Linear(no_units, 2 * self.output_dim, bias=True)
        self.Dropout = nn.Dropout(p)
        # activation to be used between hidden layers
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        KL_loss_total = 0
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.Dropout(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.Dropout(x)
        x = self.activation(x)

        # x = self.layer3(x)
        # x = self.Dropout(x)
        # x = self.activation(x)

        x = self.layer3(x)
        # x = self.activation(x)

        return x


class DNN_Hetero_MCDropout_Model_Wrapper(BaseNet):
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, p=0.5):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        # self.no_batches = no_batches
        # self.schedule = None  # [] #[50,200,400,600]
        self.network = DNN_Hetero_MCDropout_Model(input_dim=input_dim, output_dim=output_dim,
                                               no_units=no_units, p=p)
        self.network.cuda()
        self.epoch = 0
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learn_rate)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.learn_rate)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=6, factor=0.35, verbose=True, min_lr=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=no_batches, gamma=0.1 ** (1 / 500))
        self.criteria = log_gaussian_loss

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)

        # reset gradient and total loss
        self.optimizer.zero_grad()
        # fit_loss_total = 0

        # for i in range(no_samples):
        output = self.network(x)

        # calculate fit loss based on mean and standard deviation of output
        fit_loss = self.criteria(output[:, :1], y, output[:, 1:].exp(), 1)
        # print("fit_loss_total:",fit_loss_total)

        fit_loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        return fit_loss

    def valid(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)

        # for i in range(no_samples):

        output = self.network(x)

        # calculate fit loss based on mean and standard deviation of output
        valid_loss = self.criteria(output[:, :1], y, output[:, 1:].exp(), 1)
        # print("fit_loss_total:",fit_loss_total)

        return output, valid_loss

    def set_train(self):
        self.network.train()
        return

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.network.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def disable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.network.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.eval()

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()