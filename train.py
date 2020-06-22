# Lenet without deep learning framework
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

import utils
from dataset import custom_transform
from dataset.dataloader import Dataloader
from model.function import crossEntropy, grad_crossEntropy

# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='./pretrained/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--epoch', type=int, default=30, help='epoch size for training')
parser.add_argument('-f', '--freq', type=int, default=3, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--model', required=True, type=str, help='model for trainning')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--optim', default='', type=str, help='batch size for neuralNetwork')
parser.add_argument('--embed', default=300, type=int, help='embedding Size')
parser.add_argument('--dropout', default=False, type=bool, help='Dropout Implementation')

if __name__ == '__main__':
    args = parser.parse_args()
    train_transform = custom_transform.Compose([
        #        custom_transform.RandomHorizontalFlip(),
        #        custom_transform.RandomVerticalFlip(),
        #        custom_transform.ColorJitter(0.1, 0.1, 0.1, 0.1),
        #        custom_transform.RandomRotation(30),
        custom_transform.ScaleDown((64, 64)),
        custom_transform.ArrayToTensor(),
        #        custom_transform.normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # data loader
    trainloader = Dataloader(
        path=args.data_dir,
        is_train=True,
        batch_size=args.batch_size,
        transform=train_transform,
        shuffle=True,
        embed_size=args.embed,
    )

    # load network
    model = utils.check_model(args.model, args.embed, args.dropout)
    if args.pretrained is not '':
        load_path = Path(args.path)
        with open(args.pretrained, 'rb') as f:
            model = pickle.load(f)
    save_dir = Path(args.save_dir)
    result_dir = Path('result/{:s} model_Dropout_Adam_SGD_EMBED'.format(args.model))
    # for the BLEU
    plot_BLEU = utils.plot_BLEU(result_dir)
    losses = []

    # train
    for epoch in range(args.epoch):

        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (image, embed, label) in enumerate(trainloader):
            iter_start_time = time.time()
            output = model(image, embed)

            loss = list()
            grad_loss = list()
            for j in range(len(output)):
                loss.append(crossEntropy(output[j], label[j]))
                grad_loss.append(grad_crossEntropy(output[j], label[j]))

            # update parameters with backward
            grad_loss = np.stack(grad_loss, axis=0)
            model.update(grad_loss)

            # save losses for plot loss graph
            loss = np.stack(loss, axis=0)
            loss = np.mean(loss, axis=(0, 1))

            running_loss += loss
            if i % args.freq == args.freq - 1:
                losses.append(running_loss / args.freq)
                print('epoch: {}, iteration:{}/{} loss: {:0.4f}, iteration_time: {:0.4f}'.format(epoch + 1,
                                                                                                 i + 1,
                                                                                                 len(trainloader),
                                                                                                 running_loss / args.freq,
                                                                                                 time.time() - iter_start_time))
                seq_list = model.evaluate(image, trainloader)
                GT = trainloader.printGT(i)
                print('----------SEQ LIST------------')
                print(' '.join(seq_list))
                print('-------------GT --------------')
                print(' '.join(GT))
                print('------------------------------')

                running_loss = 0.0

        # print for every epoch
        print('{} epoch is end, epoch time : {:0.4f}'.format(epoch + 1, time.time() - epoch_start_time))

        # save model

        save_filename = save_dir / '{}_{}epochs.pkl'.format(args.model, epoch)
        save_lossname = save_dir / '{}_{}epochs_loss.pkl'.format(args.model, epoch)
        pickle.dump(save_lossname, open(save_filename, 'wb'))
        model.save(save_filename)
        pickle.dump(losses, open(save_lossname, 'wb'))

    print('---------------- trainning is done -----------')

'''
                utils.plot_loss(losses, '{:s} model Loss Graph'.format(args.model))
                seq_list = model.evaluate(image, trainloader)
                GT = trainloader.printGT(i)
                print(' '.join(seq_list))
                print(' '.join(GT))
                plot_BLEU(GT, seq_list)
                utils.plot_Image(result_dir, image[0], GT, seq_list, i)
'''
