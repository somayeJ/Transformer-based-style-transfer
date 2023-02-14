#### About this code:
This code computes fluency of the text and  is adapted from the code released  <a href="https://github.com/shentianxiao/language-style-transfer/blob/master/code/language_model.py"> here </a> for computing fluency. 

<!-- GETTING STARTED ## Getting Started-->

<!--*****************************1. -->

#### Running the model
* To train the model, first adjust the following parameters in the function load_arguments() of the file  ''fluency.py'':
   * --train = ../data/Yelp/train 
   * --dev = ../data/Yelp/dev 
   * --test = ../data/Yelp/test 
   * --vocab = ./save/yelp.vocab 
   * --model = ./save/model 
   * --generation_mode = '' (suffix of the file)
   
* To test the model, first adjust the following parameters in the function load_arguments() of the file  ''fluency.py'':
   * --test = ./generated_outputs/Yelp/test 
   * --vocab = ./save/yelp.vocab 
   * --model = ./save/model 
   * --load_model = True
   * --generation_mode = '.tsf' 

* Then run the following command:
   ```sh
   fluency.py
   ```
####  Dependencies
* TensorFlow 1.3.0
* Python >= 2.7
