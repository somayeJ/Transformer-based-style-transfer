#### About this code:
This code computes CPP sing n-gram based techniques  by following the same strategy proposed by the paper  <a href="https://www.aclweb.org/anthology/P19-1041/">《Disentangled Representation Learning for Non-Parallel Text Style Transfer》</a>. 

<!-- GETTING STARTED ## Getting Started-->

<!--*****************************1. -->

#### Running the code
* First adjust the parameters in the function load_arguments() of the file  ''word_overlap.py'':
   * --source_file_path = ../data/Yelp/test
   * --generated_file_path = ../outputs/test
   * --scores_output_file = ../outputs/ngram_scores 
   * --source_suffix = '' (suffix of the source file)
   * --generation_mode = '' (suffix of the generated file: '.rec' or '.tsf' or '')
   * --remove_stopwords = True (or False)

* Then run the following command:
   ```sh
   word_overlap.py
   ```
####  Dependencies
* TensorFlow 1.3.0
* Python >= 2.7
