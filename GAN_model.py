import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np


class Generator(nn.Module):
    # initializers
    def __init__(self, d, n_noise):  # 1-d vector
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(n_noise, d, bias=True)
        self.bn1 = nn.BatchNorm1d(d)
        self.linear2 = nn.Linear(d, d, bias=True)
        self.bn2 = nn.BatchNorm1d(d)
        self.linear3 = nn.Linear(d, d, bias=True)
        self.bn3 = nn.BatchNorm1d(d)

    # forward method
    def forward(self, noise):
        x = torch.tanh(self.bn1(self.linear1(noise)))
        x = torch.tanh(self.bn2(self.linear2(x)))
        x = torch.sigmoid(self.bn3(self.linear3(x)))
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(d, d, bias=True)
        self.linear2 = nn.Linear(d, 1, bias=True)

    # forward method
    def forward(self, dec):
        x = torch.tanh(self.linear1(dec))
        x = torch.sigmoid(self.linear2(x))
        return x


class GAN(object):
    def __init__(self, d, batchsize, lr, epoches, n_noise):
        self.d = d
        self.n_noise = n_noise
        self.BCE_loss = nn.BCELoss()
        self.G = Generator(self.d, self.n_noise)
        self.D = Discriminator(self.d)
        self.G.cuda()
        self.D.cuda()
        self.G_optimizer = optim.Adam(self.G.parameters(), 4*lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr)
        self.epoches = epoches
        self.batchsize = batchsize

    def train(self, pop_dec, labels, samples_pool):
        self.D.train()
        self.G.train()
        n, d = np.shape(pop_dec)
        indices = np.arange(n)

        center = np.mean(samples_pool, axis=0)
        cov = np.cov(samples_pool[:10, :].reshape((d, samples_pool[:10, :].size // d)))
        iter_no = (n + self.batchsize - 1) // self.batchsize

        for epoch in range(self.epoches):
            g_train_losses = 0

            for iteration in range(iter_no):

                # train the D with real dataset
                self.D.zero_grad()
                given_x = pop_dec[iteration * self.batchsize: (1 + iteration) * self.batchsize, :]
                given_y = labels[iteration * self.batchsize: (1 + iteration) * self.batchsize]
                batch_size = np.shape(given_x)[0]

                # (Tensor, cuda, Variable)
                given_x_ = Variable(torch.from_numpy(given_x).cuda()).float()
                given_y = Variable(torch.from_numpy(given_y).cuda()).float()
                d_results_real = self.D(given_x_.detach())

            # train the D with fake data
                fake_x = np.random.multivariate_normal(center, cov, batch_size)
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                         np.zeros((batch_size, self.d))))

                fake_y = Variable(torch.zeros((batch_size, 1)).cuda())
                fake_x_ = Variable(fake_x.cuda()).float()
                g_results = self.G(fake_x_.detach())
                d_results_fake = self.D(g_results)

                d_train_loss = self.BCE_loss(d_results_real, given_y) + \
                               self.BCE_loss(d_results_fake, fake_y)  # vanilla  GAN
                d_train_loss.backward()
                self.D_optimizer.step()

                # train the G with fake data
                self.G.zero_grad()
                fake_x = np.random.multivariate_normal(center, cov, batch_size)
                fake_x = torch.from_numpy(np.maximum(np.minimum(fake_x, np.ones((batch_size, self.d))),
                                                     np.zeros((batch_size, self.d))))
                fake_x_ = Variable(fake_x.cuda()).float()
                fake_y = Variable(torch.ones((batch_size, 1)).cuda())
                g_results = self.G(fake_x_)
                d_results = self.D(g_results)
                g_train_loss = self.BCE_loss(d_results, fake_y)   # vanilla GAN loss
                g_train_loss.backward()
                self.G_optimizer.step()
                g_train_losses += g_train_loss.cpu()
            # after each epoch, shuffle the dataset
            random.shuffle(indices)
            pop_dec = pop_dec[indices, :]

    def generate(self, sample_noises, population_size):

        self.G.eval()  # set to eval mode

        center = np.mean(sample_noises, axis=0).T  # mean value
        cov = np.cov(sample_noises.T)   # convariance
        batch_size = population_size

        noises = np.random.multivariate_normal(center, cov, batch_size)
        noises = torch.from_numpy(np.maximum(np.minimum(noises, np.ones((batch_size, self.d))),
                                                      np.zeros((batch_size, self.d))))
        decs = self.G(Variable(noises.cuda()).float()).cpu().data.numpy()
        return decs

    def discrimate(self, off):

        self.D.eval()  # set to eval mode
        batch_size = off.shape[0]
        off = off.reshape(batch_size, 1, off.shape[1])

        x = Variable(torch.from_numpy(off).cuda(), volatile=True).float()
        d_results = self.D(x).cpu().data.numpy()
        return d_results.reshape(batch_size)


