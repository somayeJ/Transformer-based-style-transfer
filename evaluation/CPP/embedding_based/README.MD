#### About this code:
This code computes Content Preservation Power (CPP) and  is adapted from the code released  <a href="https://github.com/fuzhenxin/text_style_transfer"> here </a>. 

<!-- GETTING STARTED ## Getting Started-->

<!--*****************************1. -->

#### Running the model
* To compute CPP between a source file and a generated file, first adjust the following parameters in the function load_arguments() of the file  ''glove_based_cpp.py'':
   * --file_1 = ../tmp/test 
   * --file_2 = ../data/Yelp/test 
   * --random_files = False 
   * --generation_mode = '.tsf' (suffix of file_1 (generated file))
   
* To compute CPP between a two random source files, first adjust the following parameters in the function load_arguments() of the file  ''glove_based_cpp.py'':
   * --file_1 = ../data/Yelp/test
   * --file_2 = ../data/Yelp/train 
   * --random_files = True
   * --generation_mode = '' 

* Then download pre-trained word vectors <a href="https://nlp.stanford.edu/projects/glove/"> GloVe </a> 

* Run the following command in the terminal:
   ```sh
   glove_based_cpp.py
   ```
