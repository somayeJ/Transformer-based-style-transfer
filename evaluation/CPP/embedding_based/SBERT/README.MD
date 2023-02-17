#### About this code:
This code computes Content Preservation Power (CPP) using pre-trained BERT models from  <a href=" https://www.sbert.net/docs/pretrained_cross-encoders.html"> here </a> to create the embedding vectors of the sequences. 

<!-- GETTING STARTED ## Getting Started-->

<!--*****************************1. -->

#### Running the code
* To compute CPP between a source file and a generated file, first adjust the following parameters in the function load_arguments() of the file  ''BERT_based_cpp.py'':
   * --file_1 = ../tmp/test
   * --file_2 = ../data/Yelp/test
   * --random_files = False 
   * --generation_mode = '.tsf' (suffix of file_1 (generated file))
   
* To compute CPP between  two random  files (for instance between train and test files of the corpus) , first adjust the following parameters in the function load_arguments() of the file  ''BERT_based_cpp.py'':
   * --file_1 = ../data/Yelp/test
   * --file_2 = ../data/Yelp/train
   * --random_files = True
   * --generation_mode = '' 

* Run the following command in the terminal:
   ```sh
  BERT_based_cpp.py
   ```

####  Dependencies
* Python >= 3.*
