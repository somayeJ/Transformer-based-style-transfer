import sys
import argparse
import pprint
import statistics
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
 
def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--source_file_path',
        type=str,
        default='')
    argparser.add_argument('--generated_file_path',
        type=str,
        default='')
    argparser.add_argument('--scores_output_file',
        type=str,
        default='')
    argparser.add_argument('--source_suffix',
        type=str,
        default='') #(the suffix of the source file)
    argparser.add_argument('--generation_mode',
        type=str,
        default='') #(the suffix of the generated file: '.rec' or '.tsf' or '')
    argparser.add_argument('--remove_stopwords',
            type=bool,
            default=True)
    args = argparser.parse_args()
    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')
    return args

def get_stopwords():
    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS
    all_stopwords = set()
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords
    return all_stopwords

def word_overlap_score_evaluator(src_file_path,generated_file_path,scores_file,args):
    actual_word_lists, generated_word_lists= list(), list() 
    with open(src_file_path) as source_file, open(generated_file_path) as generated_file:
        generated_lines = generated_file.readlines()
        source_lines =  source_file.readlines()
        assert len(source_lines)==len(generated_lines), "length error"
        for line_1, line_2 in zip(source_lines, generated_lines):
            actual_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_1))
            generated_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_2))

    if args.remove_stopwords:
        english_stopwords = get_stopwords()
    else:
        english_stopwords = set([]) 

    scores = list()
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        score = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)

        words_1 -= english_stopwords
        words_2 -= english_stopwords

        word_intersection = words_1 & words_2
        word_union = words_1 | words_2

        if word_union:
            score = float(len(word_intersection)) / len(word_union)
            scores.append(score)
    with open(scores_file,'w') as fw:
        for score in scores:
            fw.write(str(score))
            fw.write('\n')

    word_overlap_score = statistics.mean(scores) if scores else 0
    del english_stopwords
    return word_overlap_score

if __name__ == "__main__":
    args = load_arguments()
    src_styles = ['.neg', '.pos']
    gen_styles = ['.0', '.1']
    ngram_scores =[]
    for ss , gs in zip(src_styles, gen_styles):
        src_file_path = args.source_file_path + ss + args.source_suffix
        gen_file_path = args.generated_file_path  + args.generation_mode + gs + '.txt'
        print(src_file_path, gen_file_path)
        scores_file = args.scores_output_file + gs + '.txt'
        word_overlap_score = word_overlap_score_evaluator(src_file_path,gen_file_path,scores_file,args)
        ngram_scores.append(word_overlap_score)
    print('word_overlap_scores', ngram_scores[0],ngram_scores[1]) 
    print('Average of word_overlap_scores',float((ngram_scores[0]+ngram_scores[1])/2))

    

