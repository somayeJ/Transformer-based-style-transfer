# -*- coding: utf-8 -*-
#!//env python3
import sys
import argparse
import pprint
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--random_files',
            type=bool,
            default=False)
    argparser.add_argument('--file_1',
            type=str,
            default='')
    argparser.add_argument('--file_2',
            type=str,
            default='')
    argparser.add_argument('--generation_mode',
            type=str,
            default='') # '.rec' or '.tsf', if '', the file is a source file
    argparser.add_argument('--output',
            type=str,
            default='')
    args = argparser.parse_args()

    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')
    return args

def data_pairs(gen_dir_name, src_dir_name ,  random_files, file_gen_mode=''):
    #create suitable data: 
    #test.rec.1.txt
    #test.neg
    all_lines = []
    indexes = [".0", ".1"]
    styles = ['.neg', '.pos']
           
    for style, index_name in zip(styles, indexes):
        lines = []
        if random_files:
            file1 = gen_dir_name  + style
            file2= src_dir_name + style
        else: 
            file2= src_dir_name  + style

            if file_gen_mode == '':
                file1 = gen_dir_name  + style   
            else:
                file1 = gen_dir_name + file_gen_mode+ index_name + ".txt"
             
        with open(file1,'r') as f1,  open(file2,'r') as f2:
            lines1 = [l.strip() for l in f1.readlines()]
            lines2 = [l.strip() for l in f2.readlines()]
        for (l1,l2) in zip(lines1, lines2):
            lines.append([l1,l2])
        all_lines.append(lines)
    return all_lines

def gen_score(all_lines,  file_dir_write, model):
    res = []
    #test.rec.1.txt
    #test.neg

    for style,lines in enumerate(all_lines):# all_lines[0]: seqs tuples with the gen_style0 and src_style0, lines[1]: seqs tuples with the gen_style1 and src_style1
        w_file = file_dir_write + ".style" + str(style) + file_gen_mode+ ".txt"
        scores = model.predict(lines)
        print('done predicting lines')
        with open(w_file, 'w') as w:
            for score in scores:
                w.write(str(score)+"\n")
        res.append(np.mean(scores))
        print('done writing lines')
    return res, np.mean(res)

if __name__=="__main__":
    args = load_arguments()
    file_1_dir_name = args.file_1# gen_file
    file_2_dir_name = args.file_2# src_file
    output_dir_name_b = args.output + 'bert'
    output_dir_name_r = args.output + 'Roberta'
    file_gen_mode = args.generation_mode 
    random_files = args.random_files
    lines = data_pairs(file_1_dir_name,file_2_dir_name,random_files,file_gen_mode)
    # models from here https://www.sbert.net/docs/pretrained_cross-encoders.html 
    model_Rbert = CrossEncoder('cross-encoder/stsb-roberta-large')
    model_bert = CrossEncoder('cross-encoder/stsb-TinyBERT-L-4')

    res_b, avg_res_b = gen_score(lines, output_dir_name_b, model_bert )
    print('done model bert')

    res_r, avg_res_r = gen_score(lines, output_dir_name_r, model_Rbert )
    print('done model Robert')

    if args.random_files:
        print("Comparing random files ...")
        print(F"BERT CPP of  files {file_1_dir_name} and {file_2_dir_name} with style 1: {res_b[0]} and with style 2: {res_b[1]}")
        print(F"Average BERT CPP: {avg_res_b}")
        print('*****************************************************************')
        print(F"RoBERTa CPP of  files {file_1_dir_name} and {file_2_dir_name} with style 1: {res_r[0]} and with style 2: {res_r[1]}")
        print(F"Average RoBERTa CPP: {avg_res_r}")
    else:
        print("Comparing src & gen files ...")
        print(F"BERT CPP of  src file {file_2_dir_name} and gen file {file_1_dir_name} with style 1: {res_b[0]} and with style 2: {res_b[1]}")
        print(F"Average BERT CPP: {avg_res_b}")
        print('*****************************************************************')
        print(F"RoBERTa CPP of src file {file_2_dir_name} and gen file {file_1_dir_name} with style 1: {res_r[0]} and with style 2: {res_r[1]}")
        print(F"Average RoBERTa CPP: {avg_res_r}")
        

        

