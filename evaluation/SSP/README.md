#### About this code:
This code computes style_shift_power (SSP) and  is adapted from the code released by the paper <a href="https://arxiv.org/abs/1705.09655">《Style Transfer from Non-Parallel Text by Cross-Alignmen》</a>. 

<!-- GETTING STARTED ## Getting Started-->

<!--*****************************1. -->

#### Running the model
* To train the model, first adjust the following parameters in the function load_arguments() of the file  ''SSP_classifier.py'':
   * --train = ../../data/Yelp_small/train 
   * --dev = ../../data/Yelp_small/dev 
   * --test = ../../data/Yelp_small/test 
   * --vocab = ./yelp.vocab 
   * --model = ./model 
   * --generation_mode = '' (suffix of the file)
   
* To test the model, first adjust the following parameters in the function load_arguments() of the file  ''SSP_classifier.py'':
   * --test = ../../data/Yelp_small/test 
   * --vocab = ./yelp.vocab 
   * --model = ./model 
   * --load_model = True
   * --generation_mode = '.tsf' 

* Then run the following command:
   ```sh
   SSP_classifier.py
   ```
####  Dependencies
* TensorFlow 1.3.0
* Python >= 2.7
