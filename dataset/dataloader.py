import gzip
import math
import os
import pickle
import re
from pathlib import Path

import PIL.Image as Image
import numpy as np
from nltk import FreqDist

import dataset.text as text


# reference : PA3_data_process.ipynb
# it contains embedding process and responsible for RNN inputs and image inputs
class Dataloader():
    def __init__(self, path, is_train=True, shuffle=True, transform=None, batch_size=1, embed_size=300):
        self.path = Path(path)
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.VOCAB_SIZE = 0
        self.embed_size = embed_size
        self.descriptions = dict()

        self.train_descriptions = self.get_train_description()
        self.token = self.get_token(self.train_descriptions)
        self.MAX_LENGTH = self.get_MAX_LENGTH(self.train_descriptions)
        self.embed_Matrix = self.get_embed_Matrix(self.token)

        # train description keys
        if is_train:
            self.keys = np.array(list(self.train_descriptions.keys()))
        else:
            self.test_descriptions = self.get_test_description()
            self.keys = np.array(list(self.test_descriptions.keys()))
        self.idx = np.arange(0, len(self.train_descriptions))

        # extract first element of texts
        for ek in self.train_descriptions.keys():
            self.descriptions[ek] = [self.train_descriptions[ek][0]]

    def __len__(self):
        if self.is_train:
            n_images = math.floor(len(self.train_descriptions) / self.batch_size)
        else:
            n_images = math.floor(len(self.test_descriptions) / self.batch_size)
        return n_images

    def __iter__(self):
        return datasetIterator(self)

    def __getitem__(self, index):
        image = self.load_images(self.idx[index * self.batch_size:(index + 1) * self.batch_size])
        if self.is_train:
            embed_vectors, labels = self.load_embed_labels(
                self.idx[index * self.batch_size:(index + 1) * self.batch_size])
        else:
            embed_vectors, labels = None, None

        if self.transform is not None:
            image = self.transform(image)
        return image, embed_vectors, labels

    def get_train_description(self):
        # already exists files in the path
        path = Path(self.path)
        filename = 'train_descriptions.pkl'

        if os.path.exists(filename):
            f = open(filename, 'rb')
            train_descriptions = pickle.load(f)
            return train_descriptions

        if os.path.exists('descriptions.pkl'):
            f = open('descriptions.pkl', 'rb')
            descriptions = pickle.load(f)
        else:
            # if not then
            descriptions = dict()

            with open(path / 'Flickr8k.token.txt') as f:
                data = f.read()

            try:
                for el in data.split("\n"):
                    tokens = el.split()
                    image_id, image_desc = tokens[0], tokens[1:]

                    # dropping .jpg from image id
                    image_id = image_id.split(".")[0]

                    image_desc = " ".join(image_desc)

                    # check if image_id is already present or not
                    if image_id in descriptions:
                        descriptions[image_id].append(image_desc)
                    else:
                        descriptions[image_id] = list()
                        descriptions[image_id].append(image_desc)

            except Exception as e:
                print("Exception got :- \n", e)

            for k in descriptions.keys():
                value = descriptions[k]
                caption_list = []
                for ec in value:
                    # replaces specific and general phrases
                    sent = self.decontracted(ec)
                    sent = sent.replace('\\r', ' ')
                    sent = sent.replace('\\"', ' ')
                    sent = sent.replace('\\n', ' ')
                    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

                    # startseq is for kick starting the partial sequence generation and endseq is to stop while predicting.
                    # for more referance please check https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
                    image_cap = 'startseq ' + sent.lower() + ' endseq'
                    caption_list.append(image_cap)
                descriptions[k] = caption_list
            pickle.dump(descriptions, open("descriptions.pkl", "wb"))

        train_descriptions = dict()
        path_list = path / "Flickr_8k.trainImages.txt"
        with open(path_list, "r") as f:
            data = f.read()

        try:
            for el in data.split("\n"):
                tokens = el.split(".")
                image_id = tokens[0]
                if image_id in descriptions:
                    train_descriptions[image_id] = descriptions[image_id]

        except Exception as e:
            print("Exception got :- \n", e)
        pickle.dump(train_descriptions, open("train_descriptions.pkl", "wb"))

        return train_descriptions

    def get_test_description(self):
        # already exists files in the path
        path = Path(self.path)
        filename = 'test_descriptions.pkl'

        if os.path.exists(filename):
            f = open(filename, 'rb')
            test_descriptions = pickle.load(f)
            return test_descriptions

        if os.path.exists('descriptions.pkl'):
            f = open('descriptions.pkl', 'rb')
            descriptions = pickle.load(f)
        else:
            # if not then
            descriptions = dict()

            with open(path / 'Flickr8k.token.txt') as f:
                data = f.read()

            try:
                for el in data.split("\n"):
                    tokens = el.split()
                    image_id, image_desc = tokens[0], tokens[1:]

                    # dropping .jpg from image id
                    image_id = image_id.split(".")[0]

                    image_desc = " ".join(image_desc)

                    # check if image_id is already present or not
                    if image_id in descriptions:
                        descriptions[image_id].append(image_desc)
                    else:
                        descriptions[image_id] = list()
                        descriptions[image_id].append(image_desc)

            except Exception as e:
                print("Exception got :- \n", e)

            for k in descriptions.keys():
                value = descriptions[k]
                caption_list = []
                for ec in value:
                    # replaces specific and general phrases
                    sent = self.decontracted(ec)
                    sent = sent.replace('\\r', ' ')
                    sent = sent.replace('\\"', ' ')
                    sent = sent.replace('\\n', ' ')
                    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

                    # startseq is for kick starting the partial sequence generation and endseq is to stop while predicting.
                    # for more referance please check https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
                    image_cap = 'startseq ' + sent.lower() + ' endseq'
                    caption_list.append(image_cap)
                descriptions[k] = caption_list
            pickle.dump(descriptions, open("descriptions.pkl", "wb"))

        test_descriptions = dict()
        path_list = path / "Flickr_8k.testImages.txt"
        with open(path_list, "r") as f:
            data = f.read()

        try:
            for el in data.split("\n"):
                tokens = el.split(".")
                image_id = tokens[0]
                if image_id in descriptions:
                    test_descriptions[image_id] = descriptions[image_id]

        except Exception as e:
            print("Exception got :- \n", e)
        pickle.dump(test_descriptions, open("test_descriptions.pkl", "wb"))

        return test_descriptions

    def get_token(self, train_descriptions):
        if os.path.exists('token.pkl'):
            f = open('token.pkl', 'rb')
            token = pickle.load(f)
            token.index_word[0] = ' '
            return token

        # creating corpus
        corpus = ""

        for ec in train_descriptions.values():
            for el in ec:
                corpus += " " + el
        total_words = corpus.split()
        vocabulary = set(total_words)
        print("The size of vocablury is {}".format(len(vocabulary)))

        freq_dist = FreqDist(total_words)

        # removing least common words from vocabulary
        for ew in list(vocabulary):
            if (freq_dist[ew] < 10):
                vocabulary.remove(ew)
        self.VOCAB_SIZE = len(vocabulary) + 1
        print("Total unique words after removing less frequent word from our corpus = {}".format(self.VOCAB_SIZE))

        caption_list = []
        for el in train_descriptions.values():
            for ec in el:
                caption_list.append(ec)

        token = text.Tokenizer(num_words=self.VOCAB_SIZE)
        token.fit_on_texts(caption_list)

        pickle.dump(token, open("token.pkl", "wb"))

        # index to words are assigned according to frequency. i.e the most frequent word has index of 1
        ix_to_word = token.index_word

        for k in list(ix_to_word):
            if k >= self.VOCAB_SIZE:
                ix_to_word.pop(k, None)
        word_to_ix = dict()
        for k, v in ix_to_word.items():
            word_to_ix[v] = k

        # for exception 0 index
        token.index_word[0] = ' '
        return token

    def get_MAX_LENGTH(self, train_descriptions):
        caption_list = []
        for el in train_descriptions.values():
            for ec in el:
                caption_list.append(ec)
        print("The total caption present = {}".format(len(caption_list)))

        # finding the max_length caption
        MAX_LENGTH = 0
        temp = 0
        for ec in caption_list:
            temp = len(ec.split())
            if (MAX_LENGTH <= temp):
                MAX_LENGTH = temp
        print("Maximum caption has length of {}".format(MAX_LENGTH))
        return MAX_LENGTH

    def get_embed_Matrix(self, token):

        if os.path.exists('embedding_matrix_{}.pkl'.format(self.embed_size)):
            f = open('embedding_matrix_{}.pkl'.format(self.embed_size), 'rb')
            embed_matrix = pickle.load(f)
            return embed_matrix

        with open(self.path / 'glove.6B.{}d.txt'.format(self.embed_size), 'rb') as f:
            data = f.read()
        glove = dict()

        try:
            for el in data.decode().split("\n"):
                tokens = el.split()
                word = tokens[0]
                vec = []
                for i in range(1, len(tokens)):
                    vec.append(float(tokens[i]))
                glove[word] = vec

        except Exception as e:
            print("Exception got :- \n", e)

        EMBEDDING_SIZE = self.embed_size

        # Geti 300-dim dense vector for each of the words in vocabulary
        embedding_matrix = np.zeros((self.VOCAB_SIZE, EMBEDDING_SIZE))
        print(embedding_matrix.shape)

        # Get 300-dim dense vector for each of the words in vocabulary
        embedding_matrix = np.zeros(((self.VOCAB_SIZE), EMBEDDING_SIZE))

        word_to_ix = dict()
        for k, v in token.index_word.items():
            word_to_ix[v] = k

        for word, i in word_to_ix.items():
            embedding_vector = np.zeros(EMBEDDING_SIZE)
            if word in glove.keys():
                embedding_vector = glove[word]
                embedding_matrix[i] = embedding_vector
            else:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector

        # save the embedding matrix to file
        with open('embedding_matrix_{}.pkl'.format(self.embed_size), "wb") as f:
            pickle.dump(embedding_matrix, f)
        return embedding_matrix

    def decontracted(self, phrase):
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def load_images(self, index):
        image_list = list()
        for idx in index:
            name = self.keys[idx] + '.jpg'
            path = self.path / 'Flicker8k_Dataset' / name
            image_list.append(Image.open(path))
        return image_list

    def load_embed_labels(self, index):
        embed_list = list()
        label_list = list()
        size_list = list()
        for key in self.keys[index]:
            texts = self.descriptions[key]
            sequences = self.token.texts_to_sequences(texts)[0]
            # 뒤에 숫자 빼면 됌 만약에 한개 안빼고 싶으면
            vector = self.embed_Matrix[sequences][:-1]
            embed_list.append(vector)
            size, _ = vector.shape
            size_list.append(size)

        max_length = max(size_list)

        # get labels

        for key in self.keys[index]:
            texts = self.descriptions[key]
            # 만약 한개 안빼고 싶으면 뒤에 -1 빼면 됌
            sequences = self.token.texts_to_sequences(texts)[0][1:]
            length = len(sequences)
            sequences = np.pad(sequences, (0, max_length - length), 'constant', constant_values=(0.0, 0.0))
            lablt = np.zeros((max_length, self.token.num_words))

            lablt[np.arange(max_length), sequences] = 1.0
            lablt.astype(np.float64)
            label_list.append(lablt)

        for i, el in enumerate(embed_list):
            a, _ = el.shape
            embed_list[i] = np.pad(el, ((0, max_length - a), (0, 0)), constant_values=0.0)

        # seq_len, batch, embed_vector_size

        return np.stack(embed_list, axis=1), np.stack(label_list, axis=1)

    def printGT(self, index):
        index = self.idx[index * self.batch_size:(index + 1) * self.batch_size]
        descriptions = self.train_descriptions if self.is_train else self.test_descriptions
        texts = descriptions[self.keys[index[0]]][0]
        texts = texts.split()[1:-1]
        return texts


# for the CNNTest loader
class CNNDataloader():
    def __init__(self, path, is_train=True, shuffle=True, transform=None, batch_size=8):
        path = Path(path)
        # load image paths
        imagePath = Path(path / 'train-images-idx3-ubyte.gz') if is_train else Path(path / 't10k-images-idx3-ubyte.gz')
        labelPath = Path(path / 'train-labels-idx1-ubyte.gz') if is_train else Path(path / 't10k-labels-idx1-ubyte.gz')

        # load images
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.images = self.loadImages(imagePath)
        self.labels = self.loadLabels(labelPath)
        self.transform = transform
        self.index = 0

        # shuffle images
        self.idx = np.arange(0, self.images.shape[0])

    def __len__(self):
        n_images, _, _, _ = self.images.shape
        n_images = math.floor(n_images / self.batch_size)
        return n_images

    def __iter__(self):
        return datasetIterator(self)

    def __getitem__(self, index):
        image = self.images[self.idx[index * self.batch_size:(index + 1) * self.batch_size]]
        label = self.labels[self.idx[index * self.batch_size:(index + 1) * self.batch_size]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def loadImages(self, path):
        # with GZIP
        with gzip.open(path) as f:
            images = np.frombuffer(f.read(), 'B', offset=16)
            images = images.reshape(-1, 1, 28, 28).astype(np.float32)
            return images

    def loadLabels(self, path):
        with gzip.open(path) as f:
            labels = np.frombuffer(f.read(), 'B', offset=8)
            rows = len(labels)
            cols = labels.max() + 1
            one_hot = np.zeros((rows, cols)).astype(np.uint8)
            one_hot[np.arange(rows), labels] = 1
            one_hot = one_hot.astype(np.float64)
            return one_hot


# for RNN dataloader
class RNNDataloader():
    def __init__(self, path, embed_size=100, shuffle=True, batch_size=64):
        path = Path(path)
        self.text = open(path / 'input.txt', 'rb').read().decode(encoding='utf-8')
        print('텍스트의 길이: {}자'.format(len(self.text)))

        # 65개임
        self.vocab = sorted(set(self.text))
        self.embed = len(self.vocab)
        print('고유 문자수 {}개'.format(len(self.vocab)))

        # 고유 문자에서 인덱스로 매핑 생성
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # 텍스트 파일에서 만듬 일종의 예시 방법이죠
        # text_as_int = np.array([self.char2idx[c] for c in self.text])

        # 단일 입력에 대해 원하는 문장의 최대 길이
        # 사이즈를 뜻함
        self.seq_length = embed_size

        # load images
        self.shuffle = shuffle
        self.batch_size = batch_size

        # shuffle images
        length = len(self.text) // (self.seq_length + 3)
        self.idx = np.arange(0, length)

    def __len__(self):
        return math.floor(len(self.text) // (self.seq_length + 3) / self.batch_size)

    def __iter__(self):
        return datasetIterator(self)

    # return value (seq_length + 1, batch, VOCAB_SIZE)
    def __getitem__(self, index):
        text_list = []
        for i in self.idx[index * self.batch_size:(index + 1) * self.batch_size]:
            one_hot = np.zeros((self.seq_length + 3, len(self.vocab)))
            sentence = self.text[i * (self.seq_length + 3):(i + 1) * (self.seq_length + 3)]
            text_as_int = np.array([self.char2idx[c] for c in sentence])
            one_hot[np.arange(self.seq_length + 3), text_as_int] = 1.0
            text_list.append(one_hot)
        total = np.stack(text_list, axis=1)
        embed = total[1:-1, :, :]
        label = total[2:, :, :]
        prev = total[0, :, :]
        return prev, embed, label

    def printGT(self, index):
        i = self.idx[index * self.batch_size:(index + 1) * self.batch_size][0]
        sentence = self.text[i * (self.seq_length + 3):(i + 1) * (self.seq_length + 3)]
        return sentence


# for enumerate magic python function returns Iterator
class datasetIterator():
    def __init__(self, dataloader):
        self.index = 0
        self.dataloader = dataloader
        if dataloader.shuffle:
            np.random.shuffle(dataloader.idx)

    def __next__(self):
        if self.index < len(self.dataloader):
            item = self.dataloader[self.index]
            self.index += 1
            return item
        # end of iteration
        raise StopIteration
