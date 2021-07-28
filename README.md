## About The Repository:
This repository contains the code and data for the following paper:\
《***Name of our paper》\
The code in this folder is based on the code released by the paper <a href="https://arxiv.org/abs/1905.05621">《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》</a>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up of the repository and run the model follow these steps.
<!--*****************************1. I shoukd check the fasttext version, 2.I need to remove the perplexity requirements (kenlm) and files about pplx and add my files -->
#### Requirements 
* torchtext >= 0.4.0
* nltk
* fasttext == 0.9.3
* kenlm

#### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install the requirements
<!--*************************1. the type of the discriminator, if we want to keep or remove the multi_discriminator -->
#### Running the model
* To run the model, first adjust the following parameters in the Config class of the file  ''main.py'':
   * To train model: train = True,  dev = True
   * To test the model: test = True
   * save_path = the directory where the models and outputs are to be saved

* Then run the following command:
   ```sh
   python main.py
   ```
## Data 
* The data/yelp/ directory is the dataset used in the paper <a href="https://arxiv.org/abs/1705.09655">《Style Transfer from Non-Parallel Text by Cross-Alignmen》</a>. It is composed of the Yelp restaurant reviews.
*  The data/GYAFC/ directory is the versiion of the <a href="https://arxiv.org/abs/1803.06535"> GYAFC dataset</a> which is used in the paper《Style as Sentiment versus Style as Formality:the same or different?". 
*  Data format: Each file should consist of one sentence per line with tokens separated by a space. The two styles are represented by 0 and 1

## Dependencies
* pytorch >= 0.4.0
* Python = 3.x.x
