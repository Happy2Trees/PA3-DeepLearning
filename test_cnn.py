# Lenet without deep learning framework
# https://github.com/Site1997/LeNet-python/blob/master/LeNet.py
import argparse
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import utils
from dataset import custom_transform
from dataset.dataloader import CNNDataloader
from model import function as F

# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='result/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('-f', '--freq', type=int, default=200, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for training')
parser.add_argument('--lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_class', type=int, default=10, help='number of classes to classify of datasets')
parser.add_argument('--model', required=True, type=str, help='beta parameters for adam optimizer')
parser.add_argument('--one_class', type=int, default=2, help='to show top 3 images ')

# classes
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

if __name__ == '__main__':
    args = parser.parse_args()
    # normalize
    train_transform = custom_transform.Compose([
        custom_transform.Normalize()
    ])
    # load test loader with batch_size = 32
    testloader = CNNDataloader(
        path=args.data_dir,
        batch_size=32,
        is_train=False,
        shuffle=False
    )

    # Load pretrained models
    model = pickle.load(open('pretrained/CNNTest_1epochs_CNN', 'rb'))

    # initialize confusion matrix
    confusion_matrix = np.zeros((args.num_class, args.num_class), dtype=np.float32)

    # for top 3 class images
    img_class = []
    scores_class = []

    # 1 epochs
    for i, (image, label) in enumerate(testloader):
        output = model(image)
        # ground Truth
        gt_class = np.argmax(label, axis=-1)
        # pred class
        pred_class = np.argmax(output, axis=-1)

        # softmax
        scores = F.softmax(output)

        # for the top 3 images
        condition = (gt_class == pred_class) & (pred_class == args.one_class)
        for index in range(args.batch_size):
            if condition[index]:
                img_class.append(image[index])
                scores_class.append(scores[index, pred_class[index]])

        # make confusion matrix
        for index in range(label.shape[0]):
            confusion_matrix[gt_class[index], pred_class[index]] += 1.0

    # plot the confusion matrix
    utils.plot_confusion_matrix(confusion_matrix, classes, path=args.save_dir,
                                title='{:s} of confusion matrix'.format(args.model))

    # Total Accuracy
    print('----- total Accuracy : {:0.2f}%%'.format(100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)))

    # top 3 images
    arg = np.argsort(scores_class)[::-1]
    arg = arg[:3]

    # plot top 3 images from PA1-code
    f = 0
    fig = plt.figure(figsize=(4, 2))
    for idx in arg:
        ax = fig.add_subplot(1, 3, f + 1, xticks=[], yticks=[])
        plt.imshow(img_class[idx].reshape(28, 28))
        ax.set_title("{:0.02f}\n{:4s}".format(
            scores_class[idx] * 100, classes[args.one_class]),
            color="green")
        fig.savefig(Path(args.save_dir) / '{}_top3_scores_results.png'.format(args.model))
        f += 1
