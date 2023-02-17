# encoding=utf-8
import sys
import numpy as np
from scipy import spatial
import argparse
import pprint

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--random_files',
            type=bool, 
            default=False) # True: if 2 random files are compared
    argparser.add_argument('--file_1',
            type=str,
            default='') # generated file
    argparser.add_argument('--file_2',
            type=str,
            default='') # source file
    argparser.add_argument('--emb_dim',
            type=int,
            default=100)
    argparser.add_argument('--generation_mode',
            type=str,
            default='') # generation mode of the file_1 (generated file) : 'rec' , 'tsf', or '', ('' if file_1 is also a source file)
    argparser.add_argument('--output',
            type=str,
            default='')
    argparser.add_argument('--emb_dir',
            type=str,
            default= './glove.6B.')

    args = argparser.parse_args()

    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')

    return args


class Embedding():
    def __init__(self, dim):
        dim_all = [50, 100, 200, 300]
        assert dim in dim_all, "dim wrong"
        self.emb = self.read_emb(dim, args)
    
    def read_emb(self, dim, args):
        emb_all = dict()
        file_name = args.emb_dir +str(dim)+"d.txt"
        f = open(file_name, "r")
        lines = f.readlines()
        f.close()

        for line in lines:
            line_split = line.split()
            line_name = line_split[0]
            line_emb = line_split[1:]
            line_emb = list(map(lambda x: float(x), line_emb))
            line_emb = np.array(line_emb)
            emb_all[line_name] = line_emb 
        return emb_all
    
    def get_all_emb(self):
        return self.emb

def get_sent_emb(line, word_dict):
    zero_len_seqs=0
    line_split = line.strip().split()
    res = []
    for word in line_split:
        if word in word_dict:
            res.append(word_dict[word])
    if len(res)==0:
        res.append(word_dict["the"])
        zero_len_seqs += 1
    mm = np.mean(res, 0)
    mi = np.min(res, 0)
    ma = np.max(res, 0)
    emb = np.concatenate((mm, mi, ma))
    return emb

def com_sent(line0, line1, word_dict):
    emb0 = get_sent_emb(line0, word_dict)
    emb1 = get_sent_emb(line1, word_dict)
    result = 1 - spatial.distance.cosine(emb0, emb1)
    return result

def com_file(q_file, r_file, w_file, word_dict):
    with open(q_file, 'r') as q_file, open(r_file, 'r') as r_file, open(w_file, 'w') as w_file:
        q_file_lines = q_file.readlines()
        r_file_lines = r_file.readlines()
        len_q = len(q_file_lines)
        len_r = len(r_file_lines)

        if len_q < len_r:
            r_file_lines = r_file_lines[:len_q]
        else:
            q_file_lines = q_file_lines[:len_r]

        res = []
        for line0, line1 in zip(q_file_lines, r_file_lines):
            score = com_sent(line0, line1, word_dict)
            w_file.write(str(score)+"\n")
            res.append(score)
    return res

def gen_score(gen_dir_name, src_dir_name,  file_dir_write, emb_dim,file_gen_mode=''):
    emb = Embedding(emb_dim)
    word_dict = emb.get_all_emb()
    res = []
    indexes = [".0", ".1"]
    styles = ['.neg', '.pos']

    for style, index_name in zip(styles, indexes):
        gen_file = gen_dir_name + file_gen_mode + index_name + '.txt'
        src_file = src_dir_name  + style
        w_file = file_dir_write + ".style" + index_name + file_gen_mode + ".txt"
        scores = com_file(gen_file, src_file, w_file, word_dict)
        res.append(np.mean(scores))    
    avg_res = np.mean(res)
    return res, avg_res

def gen_score_random(file_name_1, file_name_2, file_dir_write,emb_dim, file_gen_mode=''):
    emb = Embedding(emb_dim)
    word_dict = emb.get_all_emb()
    res = []
    styles = ['.neg', '.pos']
    for style in styles:
        file_1 = file_name_1 + style
        file_2 = file_name_2 + style
        w_file = file_dir_write + ".random" + style + ".semantics.txt"
        scores = com_file(file_1, file_2, w_file, word_dict)
        res.append(np.mean(scores))
    avg_res = np.mean(res)
    return res, avg_res

if __name__=="__main__":
    args = load_arguments()
    file_1_dir_name = args.file_1
    file_2_dir_name = args.file_2
    output_dir_name = args.output
    file_gen_mode = args.generation_mode 
    emb_dim =  args.emb_dim
    print('Computing Content Preservation Power (CPP) ...')
    if args.random_files:
        print("Comparing random files ...")
        res, avg_res = gen_score_random(file_1_dir_name, file_2_dir_name, output_dir_name,emb_dim) 
        print("CPP  (random files with style 1):", res[0])
        print("CPP  (random files with style 2):", res[1])
        print("Average CPP:", avg_res)
    else:
        print("Comparing source and generated files ...")
        res, avg_res = gen_score(file_1_dir_name, file_2_dir_name, output_dir_name, emb_dim, file_gen_mode) 
        print("CPP (between files with style 1)",res[0])
        print("CPP (between files with style 2)",res[1])
        print("Average CPP:", avg_res)