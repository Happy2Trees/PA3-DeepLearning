from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from model.models import ImageCaptionLSTM, ImageCaptionRNN, CNNTest, RNNTest, LSTMTest


# return to corresponding model
def check_model(model, embed=300, dropout=False):
    models = ['LSTM', 'RNN']
    if model == 'LSTM':
        return ImageCaptionLSTM(embed, dropout=dropout)
    elif model == 'RNN':
        return ImageCaptionRNN(embed, dropout=dropout)
    elif model == 'CNNTest':
        return CNNTest()
    elif model == 'RNNTest':
        return RNNTest(65)
    elif model == 'LSTMTest':
        return LSTMTest(65)
    else:
        return ImageCaptionRNN(embed, dropout=dropout)


# draw loss graph
def plot_loss(losses, title):
    plt.figure(0)
    plt.plot(losses, label='trainning losses')
    plt.xlabel("number of iteration")
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(title + '.png')


#    plt.show()


# https://stackoverflow.com/questions/56968434/bleu-score-in-python-from-scratch
class plot_BLEU(object):
    def __init__(self, path):
        self.path = Path(path)
        self.score1 = list()
        self.score2 = list()
        self.gt = None
        self.pred = None

    def __call__(self, gt, pred):
        self.score1.append(self.bleu_score1(gt, pred))
        self.score2.append(self.bleu_score2(gt, pred))

    def draw_plot(self):
        plt.figure(1)
        plt.title('BLEU-Score')
        plt.plot(self.score1, label='BLEU-1')
        plt.plot(self.score2, label='BLEU-2')

        Path(self.path).mkdir(parents=True, exist_ok=True)

        plt.savefig(self.path / 'BLEU_Score.png')
        plt.show()

    def bleu_score1(self, original, machine_translated):
        precision = 0
        for index, x in enumerate(original):
            if machine_translated[index] == x:
                precision += 1
        min_d = min(1, len(machine_translated) / len(original))

        return min_d * (precision / len(original))

    def bleu_score2(self, original, machine_translated):
        precision1 = 0
        precision2 = 0
        for index, x in enumerate(original[:-1]):
            if machine_translated[index] == x:
                precision1 += 1
                if machine_translated[index + 1] == original[index + 1]:
                    precision2 += 1

        min_d = min(1, len(machine_translated) / len(original))
        return min_d * np.power((precision1 / len(original)) * (precision2 / len(original[:-1])), 1 / 2)


'''  NLTK Library

        gt = [gt]
        self.score1.append(sentence_bleu(gt, pred, weights=(1, 0, 0, 0)))
        self.score2.append(sentence_bleu(gt, pred, weights=(1, 1, 0, 0)))

        plt.figure(1)
        plt.title('BLEU-Score')
        plt.plot(self.score1, label='BLEU-1')
        plt.plot(self.score2, label='BLEU-2')


        Path(self.path).mkdir(parents=True, exist_ok=True)

        plt.savefig(self.path/'BLEU_Score.png')
        plt.show()
'''


# plot Images
def plot_Image(path, image, gt, pred, index):
    path = Path(path)
    image = image.transpose(1, 2, 0)
    gt = ' '.join(gt)
    pred = ' '.join(pred)

    plt.figure(2, figsize=(10, 10))
    plt.imshow(image)
    plt.title('Ground Truth : {:s}'.format(gt))
    plt.xlabel('Prediction : {:s}'.format(pred))

    # make folder and create testImages
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path / '{:5d}iter-testImage.png'.format(index))
    # plt.show()


# reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, target_names, path, title='Confusion matrix'):
    accuracy = np.trace(cm) / np.sum(cm).astype(np.float64)

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks)
        plt.yticks(tick_marks, target_names)

    # classwise accuracy text
    cm = cm.astype('float64') / cm.sum(axis=1)

    # threshold values
    thresh = cm.max() * (2 / 3)
    # put the accuracy each matrix cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # classwise accuracy !
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground Truth')
    plt.xlabel('Output label\naccuracy={:0.4f} -- each text = Classwise Accuracy--'.format(accuracy))
    plt.savefig(Path(path) / title)
    plt.show()
