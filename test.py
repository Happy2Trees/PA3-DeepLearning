# Lenet without deep learning framework
import argparse
import pickle
import time
from pathlib import Path

import utils
from dataset import custom_transform
from dataset.dataloader import Dataloader

# for inputs argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets', metavar='PATH', help='path of training directory')
parser.add_argument('--save_dir', type=str, default='./pretrained/', metavar='PATH',
                    help='path of saving checkpoint pth')
parser.add_argument('--pretrained', type=str, required=True, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--loss', type=str, required=True, default='', metavar='PATH', help='load pretrained weights')
parser.add_argument('--epoch', type=int, default=30, help='epoch size for training')
parser.add_argument('-f', '--freq', type=int, default=100, help='print frequency for training or test')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--model', required=True, type=str, help='model for trainning')
parser.add_argument('--embed', default=300, type=int, help='embedding Size')

if __name__ == '__main__':
    args = parser.parse_args()
    train_transform = custom_transform.Compose([
        custom_transform.RandomHorizontalFlip(),
        custom_transform.RandomVerticalFlip(),
        custom_transform.ColorJitter(0.1, 0.1, 0.1, 0.1),
        custom_transform.RandomRotation(30),
        custom_transform.ScaleDown((64, 64)),
        custom_transform.ArrayToTensor(),
        #        custom_transform.normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # data loader
    testloader = Dataloader(
        path=args.data_dir,
        batch_size=args.batch_size,
        transform=train_transform,
        is_train=False,
        shuffle=False,
        embed_size=args.embed,
    )

    # load network
    with open(args.pretrained, 'rb') as f:
        model = pickle.load(f)

    result_dir = Path('result/{:s}model_Dropout_Adam_SGD_EMBED'.format(args.model))
    # for the BLEU
    plot_BLEU = utils.plot_BLEU(result_dir)

    # draw losses graph
    losses = pickle.load(open(args.loss, 'rb'))
    utils.plot_loss(losses, '{:s} model Loss Graph'.format(args.model))

    # test

    epoch_start_time = time.time()

    for i, (image, _, _) in enumerate(testloader):
        iter_start_time = time.time()
        seq_list = model.evaluate(image, testloader)
        GT = testloader.printGT(i)
        plot_BLEU(GT, seq_list)
        # evaluate loss
        if i % args.freq == args.freq - 1:
            utils.plot_Image(result_dir, image[0], GT, seq_list, i)
            print('iteration:{}/{}  iteration_time: {:0.4f}'.format(i + 1, len(testloader),
                                                                    time.time() - iter_start_time))

    plot_BLEU.draw_plot()

    print('---------------- test is done -----------')
