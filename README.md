## About The Repository:
This repository contains the code and data for the following paper:\
《Local or Global:The Variation in the Encoding of Style Across Sentiment and Formality》\
The code in this folder is based on the code released by the paper <a href="https://arxiv.org/abs/1905.05621">《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》</a>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up of the repository and run the model follow these steps.
<!--*****************************1. I shoukd check the fasttext version, 2.I need to remove the perplexity requirements (kenlm) and files about pplx and add my files -->
#### Requirements 
* torchtext >= 0.4.0

#### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/somayeJ/Transformer-based-style-transfer.git
   ```
2. Install the requirements
<!--*************************1. the type of the discriminator, if we want to keep or remove the multi_discriminator -->
#### Running the model
* To train the model, first adjust the following parameters in the Config class of the file  ''main.py'':
	* self.train = True
   * self.dev = True
   * self.test = False
	* self.best_model_path = ''
   * self.save_path = the directory to save the outputs
   
* To test the model, first adjust the following parameters in the Config class of the file  ''main.py'':
	* self.train = False
   * self.dev = False
   * self.test = True
	* self.best_model_path = self.save_path + '_num_dev_best_model'  (where _num_ should be replaced with the biggest no of the saved models)
   * self.save_path = the directory to save the outputs 

* Then run the following command:
   ```sh
   python main.py
   ```
## Data 
* The data/yelp_large/ directory contains the  Yelp restaurant reviews dataset used in the paper <a href="https://arxiv.org/abs/1705.09655">《Style Transfer from Non-Parallel Text by Cross-Alignmen》</a>. 
*  The data/yelp_small/ directory contains the  Yelp restaurant reviews dataset used in the paper <a href="https://arxiv.org/abs/1905.05621">《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》</a>
*  Data format: Each file should consist of one sentence per line with tokens separated by a space. The two styles are represented by 0 and 1

## Dependencies
* pytorch >= 0.4.0
* Python = 3.x.x
