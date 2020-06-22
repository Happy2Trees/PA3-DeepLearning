# Lenet without deep learning framework
# https://github.com/Site1997/LeNet-python/blob/master/LeNet.py
import argparse
import time
from pathlib import Path

import numpy as np

import model.models as models
import utils
from dataset import custom_transform
from dataset.dataloader import CNNDataloader
from model.function import crossEntropy, grad_crossEntropy

# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='pretrained/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--epoch', type=int, default=30, help='epoch size for training')
parser.add_argument('-f', '--freq', type=int, default=10, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--model', required=True, type=str, help='model for trainning')

if __name__ == '__main__':
    args = parser.parse_args()
    train_transform = custom_transform.Compose([
        #        custom_transform.ColorJitter(0.1, 0.1, 0.1, 0.1),
        #        custom_transform.RandomRotation(30),
        #        custom_transform.ScaleDown((64, 64)),
        #        custom_transform.ArrayToTensor(),
        #        custom_transform.normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        custom_transform.Normalize()
    ])
    # data loader
    trainloader = CNNDataloader(
        path=args.data_dir,
        is_train=True,
        batch_size=args.batch_size,
        transform=train_transform,
        shuffle=True
    )

    # load network
    model = models.CNNTest()

    save_dir = Path(args.save_dir)
    losses = []
    # train
    for epoch in range(args.epoch):

        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (image, label) in enumerate(trainloader):
            iter_start_time = time.time()
            output = model(image)

            loss = crossEntropy(output, label)
            grad_loss = grad_crossEntropy(output, label)

            # update parameters with backward
            model.update(grad_loss)

            # save losses for plot loss graph
            loss = np.mean(loss, axis=0)

            running_loss += loss
            if i % args.freq == args.freq - 1:
                losses.append(running_loss / args.freq)
                utils.plot_loss(losses, '{:s} model Loss Graph'.format('CNNTest'))
                print('epoch: {}, iteration:{}/{} loss: {:0.4f}, iteration_time: {:0.4f}'.format(epoch + 1,
                                                                                                 i + 1,
                                                                                                 len(trainloader),
                                                                                                 running_loss / args.freq,
                                                                                                 time.time() - iter_start_time))
                running_loss = 0.0

        # print for every epoch
        print('{} epoch is end, epoch time : {:0.4f}'.format(epoch + 1, time.time() - epoch_start_time))

        # save model
        save_filename = save_dir / '{}_{}epochs_CNN'.format(args.model, epoch)
        model.save(save_filename)
