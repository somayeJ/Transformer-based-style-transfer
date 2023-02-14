import os
import sys
import argparse
import pprint
import pandas as pd
import random
import time
import numpy as np
import tensorflow as tf
from vocab import Vocabulary, build_vocab
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--generation_mode',
            type=str,
            default='') # '.tsf' or '.rec' or ''
    argparser.add_argument('--train',
            type=str,
            default='')
    argparser.add_argument('--dev',
            type=str,
            default='')
    argparser.add_argument('--test',
            type=str,
            default='')
    argparser.add_argument('--online_testing',
            type=bool,
            default=False)
    argparser.add_argument('--vocab',
            type=str,
            default='')
    argparser.add_argument('--embedding',
            type=str,
            default='')
    argparser.add_argument('--model',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            type=bool,
            default=False)
    argparser.add_argument('--batch_size',
            type=int,
            default=64)
    argparser.add_argument('--max_epochs',
            type=int,
            default=20)
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=1000)
    argparser.add_argument('--max_train_size',
            type=int,
            default=-1)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.5)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)
    argparser.add_argument('--shuffle_sentences',
        type=bool,
        default=False) # True if calculating lower_bound of fluency
    args = argparser.parse_args()

    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')
    return args


def load_sent(path, max_size=-1):

    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    print(data[:4])
    return data

def load_sent_shuffle(path, max_size=-1):
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            l_split = line.split()
            random.shuffle(l_split)
            random.shuffle(l_split)
            data.append(l_split)
        print(data[0])
    return data

def create_cell(dim, n_layers, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim, reuse=tf.AUTO_REUSE )
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    if n_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    return cell

class Model(object):

    def __init__(self, args, vocab):
        dim_emb = args.dim_emb
        dim_z = args.dim_z
        n_layers = args.n_layers
        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.inputs = tf.placeholder(tf.int32, [None, None],  # batch_size * max_len
            name='inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')
        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_z, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        cell = create_cell(dim_z, n_layers, self.dropout)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
            dtype=tf.float32, scope='language_model') # outputs, The RNN output Tensors
        outputs = tf.nn.dropout(outputs, self.dropout)
        outputs = tf.reshape(outputs, [-1, dim_z])
        self.logits = tf.matmul(outputs, proj_W) + proj_b

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=self.logits)
        loss *= tf.reshape(self.weights, [-1]) # hazf padding_token effects
        self.tot_loss = tf.reduce_sum(loss) # jamee tamame maghadire tamame jomleha baraye yek batch
        self.sent_loss = self.tot_loss / tf.to_float(self.batch_size) # loss har jomle

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.sent_loss)

        self.saver = tf.train.Saver()

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print('Loading model from', args.model)
        model.saver.restore(sess, args.model)
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    return model

def get_lm_batches(x, word2id, batch_size):
    '''
    # we pad sentences to the max(len) in a batch and put the pad_tokens at the end of the seq
    :param x: all the data (training data for example)
    :param word2id:
    :param batch_size:
    :return: a list of (len(x)/batch_size) ta dicts, in each dict there are {'inputs': go_x,
                        'targets': x_eos,
                        'weights': weights,
                        'size': t-s} and the size of each of the dictionaries is the batch size
    '''
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    x = sorted(x, key=lambda i: len(i))

    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))

        go_x, x_eos, weights = [], [], []
        max_len = max([len(sent) for sent in x[s:t]])
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            l = len(sent)
            padding = [pad] * (max_len - l)
            go_x.append([go] + sent_id + padding)
            x_eos.append(sent_id + [eos] + padding)
            weights.append([1.0] * l + [0.0] * (max_len-l+1))


        batches.append({'inputs': go_x,
                        'targets': x_eos,
                        'weights': weights,
                        'size': t-s })# size: size of the batch
        s = t

    return batches

def evaluate(sess, args, vocab, model, x):
    '''
    :param sess:
    :param args:
    :param vocab:
    :param model:
    :param x:
    :return:
    '''
    batches = get_lm_batches(x, vocab.word2id, args.batch_size)
    tot_loss, n_words = 0, 0
    loss=[]
    loss_sents=[]
    tot_loss =0
    for batch in batches:        
        tot_loss0 = sess.run(model.tot_loss,
            feed_dict={model.batch_size: batch['size'],
                       model.inputs: batch['inputs'],
                       model.targets: batch['targets'],
                       model.weights: batch['weights'],
                       model.dropout: 1}) # tot_loss = loss dar yek batch hast!
        n_words += np.sum(batch['weights']) # tedad kalamat dakhele yek batch hast
        tot_loss += tot_loss0
    return np.exp(tot_loss / n_words), loss, loss_sents

if __name__ == '__main__':
    args = load_arguments()
    suffix= args.generation_mode 
    if args.train:
        train0 = load_sent(args.train  + '.neg') 
        train1 = load_sent(args.train  + '.pos')

        train = train0 + train1
        if not os.path.isfile(args.vocab):
            build_vocab(train, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print('vocabulary size', vocab.size)

    if args.dev:
        dev0 = load_sent(args.dev  + '.neg') # 
        dev1 = load_sent(args.dev  + '.pos')

        dev = dev0 + dev1

    if args.test:
        if  args.shuffle_sentences: 
            if suffix == '':
                test0 = load_sent_shuffle(args.test +'.neg')
                test1 = load_sent_shuffle(args.test +'.pos')
            else:
                test0 = load_sent_shuffle(args.test + suffix +'.0.txt')
                test1 = load_sent_shuffle(args.test + suffix +'.1.txt')
        else:
            if suffix == '':
                test0 = load_sent(args.test +'.neg')
                test1 = load_sent(args.test +'.pos')
            else:
                test0 = load_sent(args.test + suffix +'.0.txt')
                test1 = load_sent(args.test + suffix +'.1.txt')
        test = test0 + test1

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)
        if args.train:
            batches = get_lm_batches(train, vocab.word2id, args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            loss = 0.0
            best_dev = float('inf')
            learning_rate = args.learning_rate

            for epoch in range(args.max_epochs):
                print('----------------------------------------------------')
                print('epoch %d, learning_rate %f' % (epoch + 1, learning_rate))

                for batch in batches:
                    step_loss, _ = sess.run([model.sent_loss, model.optimizer],
                        feed_dict={model.batch_size: batch['size'],
                                   model.inputs: batch['inputs'],
                                   model.targets: batch['targets'],
                                   model.weights: batch['weights'],
                                   model.dropout: args.dropout_keep_prob,
                                   model.learning_rate: learning_rate})

                    step += 1
                    loss += step_loss / args.steps_per_checkpoint

                    if step % args.steps_per_checkpoint == 0:
                        print('step %d, time %.0fs, loss %.2f' \
                            % (step, time.time() - start_time, loss))
                        loss = 0.0

                if args.dev:
                    ppl,ll,ss = evaluate(sess, args, vocab, model, dev)
                    print ('dev perplexity %.2f' % ppl)
                    if ppl < best_dev:
                        best_dev = ppl
                        unchanged = 0
                        print ('Saving model...')
                        model.saver.save(sess, args.model)

        if args.test:
            ppl,losss, sents = evaluate(sess, args, vocab, model, test)
            print(len(losss))
            print("**************************************************************")
            print(len(sents))
            print('test perplexity %.2f' % ppl)
