import torch
import time
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, test, auto_eval

class Config():
    def __init__(self):
        self.test = True
        self.train = False
        self.dev = False

        self.data_path = './data/yelp/'
        self.min_freq = 5 # 
        self.max_length =  16

        self.best_model_path='' # the address to the best saved model, which is the model with the highest number
        self.log_dir = 'runs/exp_cond_yelp' 
        self.save_path = './save/yelp/' # test

        self.cyc_rec_enable = True
        self.slf_factor = 1
        self.cyc_factor = 0.5
        self.adv_factor = 1

        self.inp_shuffle_len = 0
        self.inp_unk_drop_fac = 0
        self.inp_rand_drop_fac = 0
        self.inp_drop_prob = 0.0

        self.discriminator_gold_train = True
        self.n_epochs = 20
        self.batch_size = 64
        self.lr_F = 0.0001
        self.lr_D = 0.0001
        self.log_steps = 1000 

        self.learned_pos_embed = True
        self.dropout = 0
        self.drop_rate_config = [(1, 0)]
        self.temperature_config = [(1, 0)]
        self.pretrained_embed_path = '../data/embedding/'
        self.device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        self.load_pretrained_embed = False
        self.embed_size = 256
        self.d_model = 256
        self.h = 4
        self.num_styles = 2
        self.num_classes = 2
        self.num_layers = 4
        self.L2 = 0
        self.iter_D = 10
        self.iter_F = 5

    def n_batch_iter(self):# n_training_steps(n_global_steps) which shows the no of batches that iterate to have the specfied no of epochs
        file_names=['train.pos', 'train.neg']
        with open(self.data_path + 'train.pos') as f_pos, open(self.data_path  + 'train.neg') as f_neg:
            data_p = f_pos.readlines()
            data_n = f_neg.readlines()
            if len(data_p) > len(data_n):
                big_len = len(data_p)
            else:
                big_len = len(data_n)
            n = big_len/self.batch_size # no of batches(batch_ietrations) in 1 epoch
            if  n>int(n):
                n = int(n)+1 
            self.eval_steps = n
            return  int(n * self.n_epochs)

def main():
    config = Config()

    n_global_steps = config.n_batch_iter() # no of batches(batch_ietrations) in 1 batch * no_epoch
    print('eval_steps', config.eval_steps)
    print('n_global_steps',n_global_steps)
    print('**********************************************************************')
    print('slf_factor', config.slf_factor)
    print('cyc_factor', config.cyc_factor)
    print('adv_factor',config.adv_factor)
    print('cyc_rec_enable',config.cyc_rec_enable)
 
    print('**********************************************************************')
    print('slf_test', config.test)
    print('self_train', config.train)
    print('self_dev', config.train)
    print('data_path',config.data_path)
    print('save.path',config.save_path)
    print('discriminator_gold_train',config.discriminator_gold_train)
    print('max_length', config.max_length)
    print('**********************************************************************')
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    print('Vocab size:', len(vocab))

    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    
    if config.train:
        train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters,n_global_steps)
    elif  config.test:
        test(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    else:
        print('Either train or test model should be specified as True in configuartion')
        input()

if __name__ == '__main__':
    main()
