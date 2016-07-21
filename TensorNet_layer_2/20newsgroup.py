#!/usr/bin/env python

from __future__ import print_function

import sys
import os
from os.path import exists, join
import time
import cPickle
import optparse

import numpy as np
import scipy.sparse as Sp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sktensor import dtensor, cp_als
import theano
import theano.tensor as T
import lasagne

from ttlayer import TTLayer

'''
# tuning good hyperparameter using all category
training all:
$ python 20newsgroup.py -e 10000 > all.txt

training domain A(src)
# save trained params to A.pkl
$ python 20newsgroup.py -r A -d A.pkl -e 10000 > A.txt
training domain B(tgt)
# load trained params from A.pkl
$ python 20newsgroup.py -r B -l A.pkl -d B.pkl -e 10000 > B.txt
# load low-rank with R rank1 from A.pkl
$ python 20newsgroup.py -r B -l A.pkl -d B_1.pkl -e 10000 -R 1 > B_1.txt

A.txt, B.txt will contain training information(training error, validation error, validation accuracy)
$ python plot.py -i A.txt -o A
$ python plot.py -i B.txt -o B
'''

def parse_arg():
    parser = optparse.OptionParser('usage%prog [-l load parameterf from] [-d dump parameter to] [-e epoch] [-r src or tgt]')
    parser.add_option('-l', dest='fin')
    parser.add_option('-d', dest='fout')
    parser.add_option('-e', dest='epoch')
    parser.add_option('-r', dest='A_B')
    parser.add_option('-R', dest='rank')
    (options, args) = parser.parse_args()
    return options

# src
A = np.array(range(10))
# tgt
B = 10+A

BATCH_EPOCH = 20
ROWN, BATCHN = None, None
HashN = 200
isDownload = True
covDone = True
TARGET  = ['dense_2']

def load_dataset(A_B):
    def download_save_by_category():
        global HashN
        # train
        newsgroups = fetch_20newsgroups(subset='train')
        first = True
        TfIdfVec = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
        vectors = TfIdfVec.fit_transform(newsgroups.data) # in scipy.sparse.csr_matrix format
        dcmp = TruncatedSVD(n_components=HashN) # sparse linear discriminant analysis
        vectors = dcmp.fit_transform(vectors, newsgroups.target)
        target = newsgroups.target
        print('[LDA] transformed dimension={0}'.format(vectors.shape[1]))
        HashN = vectors.shape[1]

        if not exists('train'):
            os.makedirs('train')

        for i in range(20):
            x = []
            for j in range(len(target)):
                if target[j] == i:
                    x.append(vectors[j])
            cPickle.dump(x, open(join('train', '{0}.pkl'.format(i)), 'w+'))
        # test
        newsgroups = fetch_20newsgroups(subset='test')
        vectors = TfIdfVec.transform(newsgroups.data)
        vectors = dcmp.transform(vectors)
        target = newsgroups.target
        if not exists('test'):
            os.makedirs('test')

        for i in range(20):
            x = []
            for j in range(len(target)):
                if target[j] == i:
                    x.append(vectors[j])
            cPickle.dump(x, open(join('test', '{0}.pkl'.format(i)), 'w+'))

    def read_and_split(filepath, digit, NUM=None, Split=True):
        data = cPickle.load(open(filepath, 'r'))
        # instance number to use
        if NUM is not None:
            data = data[:NUM]
        target = [digit for i in range(len(data))]
        if not Split:
            return Sp.csr_matrix(data), target
        # split train/valid
        valid_ratio = 0.2
        split = int(len(target)*valid_ratio)
        valid = data[:split]
        train = data[split:]
        valid_tgt = target[:split]
        train_tgt = target[split:]
        return Sp.csr_matrix(train), Sp.csr_matrix(valid), train_tgt, valid_tgt

    def get_classes(classes):
        train, valid, test, train_tgt, valid_tgt, test_tgt = None, None, None, None, None, None
        for digit in classes:
            tr, v, trt, vt = read_and_split('train/{0}.pkl'.format(digit), digit)
            te, tet = read_and_split('test/{0}.pkl'.format(digit), digit, Split=False)
            if train is None:
                train, train_tgt = (Sp.vstack(tr), trt) if tr is not None and len(trt)>0 else (train, train_tgt)
            else:
                train, train_tgt = (Sp.vstack([train, Sp.vstack(tr)]), train_tgt+trt) if tr is not None and len(trt)>0 else (train, train_tgt)
            if valid is None:
                valid, valid_tgt = (Sp.vstack(v), vt) if v is not None and len(vt)>0 else (valid, valid_tgt)
            else:
                valid, valid_tgt = (Sp.vstack([valid, Sp.vstack(v)]), valid_tgt+vt) if v is not None and len(vt)>0 else (valid, valid_tgt)
            if test is None:
                test, test_tgt = (Sp.vstack(te), tet) if te is not None and len(tet)>0 else (test, test_tgt)
            else:
                test, test_tgt = (Sp.vstack([test, Sp.vstack(te)]), test_tgt+tet) if te is not None and len(tet)>0 else (test, test_tgt)
        global ROWN
        ROWN = len(train_tgt)+len(valid_tgt)+len(test_tgt)
        global BATCHN
        BATCHN = ROWN / BATCH_EPOCH
        return train, valid, test, np.array(train_tgt), np.array(valid_tgt), np.array(test_tgt)

    if not isDownload:
        download_save_by_category()
    if not covDone:
        print('calculating transform matrix for covariance')
        calcAcov()
    classes = A if A_B == 'A' else B
    if A_B is None:
        classes = range(20)
    return get_classes(classes)


# ##################### Build the neural network model #####################
def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, HashN),
                                     input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, name='dense_1', num_units=150,
            nonlinearity=None)
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, name='dense_2', num_units=100,
            nonlinearity=None)
    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2, name='dense_3', num_units=50,
            nonlinearity=None)
    l_hid4 = lasagne.layers.DenseLayer(
            l_hid3, name='dense_4', num_units=50,
            nonlinearity=None)
    l_hid5 = lasagne.layers.DenseLayer(
            l_hid4, name='dense_5', num_units=50,
            nonlinearity=None)
    l_out = lasagne.layers.DenseLayer(
            l_hid5, num_units=20,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

# CP low-rank approximate
def approx_CP_R(value, R):
    if value.ndim < 2:
        return value
    T = dtensor(value)
    P, fit, itr, exetimes = cp_als(T, R, init='random')
    Y = None
    for i in range(R):
        y = P.lmbda[i]
        o = None
        for l in range(T.ndim):
            o = P.U[l][:,i] if o is None else np.outer(o, P.U[l][:,i])
        y = y * o
        Y = y if Y is None else Y+y
    return Y

def load_largest_rank(A, O, rank):
    def all_layers(A, O, TARGET):
        AA, BB = [], []
        row, column = -1, -1
        for i in range(len(A)):
            if A[i].name not in TARGET:
                continue
            if 'dense' in A[i].name or 'conv' in A[i].name:
                w, b = A[i].get_params()
                w, b = w.get_value(), b.get_value()
                print('w={},b={}'.format(w.shape, b.shape))
                BB.append(b)
                c = 1
                for j in range(1,w.ndim):
                    c*=w.shape[j]
                T = w.reshape((w.shape[0],c))
                AA.append(T)
                row, column = max(row, len(T)), max(column, len(T[1]))
            else:
                print('layers other than "dense", "conv" layers, donnot have params')

        for i in range(len(AA)):
            ac = len(AA[i][0])
            while row > len(AA[i]):
                AA[i] = np.append(AA[i], np.zeros((1,ac)), axis=0)
            if column > ac:
                TAA = []
                for j in range(row):
                    TAA += [np.append(AA[i][j], np.zeros(column-ac))]
                AA[i] = np.array(TAA)
        for i in range(len(BB)):
            ac = len(BB[i])
            if column > ac:
                BB[i] = np.append(BB[i], np.zeros(column-ac))
        return np.array(AA), np.array(BB)
    def de_all_layers(O, AA, BB, TARGET):
        A = []
        ai = 0
        for i in range(len(O)):
            if O[i].name not in TARGET:
                x = O[i].get_params()
                if x != []:
                    w, b = x
                    A.append(w.get_value().astype(np.float32))
                    A.append(b.get_value().astype(np.float32))
            else:
                if O[i].get_params() == []: # maxpool, dropout donnot have params
                    continue
                w, b = O[i].get_params()
                w, b = w.get_value(), b.get_value()
                c = 1
                for j in range(1,w.ndim):
                    c*=w.shape[j]
                TAA = []
                X = AA[ai]
                X = X[:w.shape[0]]
                for j in range(X.shape[0]):
                    TAA += [X[j][:c]]
                A.append(np.array(TAA).reshape((w.shape)).astype(np.float32))
                A.append(BB[ai][:len(b)].astype(np.float32))
                ai += 1
        return A
    ### tensorization ###
    TAA, TBB = all_layers(A, O, TARGET)
    print('decomposing tensor W of shape {}...'.format(TAA.shape))
    print('decomposing tensor B of shape {}...'.format(TBB.shape))
    AA = approx_CP_R(TAA, int(rank)).reshape(TAA.shape)
    BB = approx_CP_R(TBB, int(rank)).reshape(TBB.shape)
    ### de-tensorization ###
    # all layers
    A = de_all_layers(O, AA, BB, TARGET)
    return A
def load_smallest_rank(A, O, rank):
    TA = load_largest_rank(A, O, rank)
    return np.array(A)-np.array(TA)+np.array(O)

# ############################# Batch iterator #################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize] if shuffle else slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ##################
def main(num_epochs=500,fin_params=None,fout_params=None,A_B=None, rank=None):
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(A_B)
    X_train, X_val, X_test = X_train.todense(), X_val.todense(), X_test.todense()
    print('#train = {0}, #test = {1}, #valid = {2}'.format(len(y_train), len(y_test), len(y_val)))
    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    # Create neural network model.
    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    # load parameters
    if fin_params is not None:
        A = cPickle.load(open(fin_params, 'r'))
        O = lasagne.layers.get_all_layers(network)
        if rank is not None:
            # try largest 1-rank approximate
            A = load_largest_rank(A, O, rank)
            # try largest 1-rank approximate
            # A = load_smallest_rank(A, O, rank)
        lasagne.layers.set_all_param_values(network, A)
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=1e-3, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCHN, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCHN, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, BATCHN, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # dump parameters
    cPickle.dump(lasagne.layers.get_all_layers(network), open(fout_params, 'w+'))


if __name__ == '__main__':
    opts = parse_arg()
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(opts.epoch)
        kwargs['fin_params'] = opts.fin
        kwargs['fout_params'] = opts.fout
        kwargs['A_B'] = opts.A_B
        kwargs['rank'] = opts.rank

    main(**kwargs)
