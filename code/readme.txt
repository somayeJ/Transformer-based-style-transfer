This folder contains the code and Yelp corpus of the paper.
____________________________________________________________________________________
Requirements:
	pytorch >= 0.4.0
	torchtext >= 0.4.0
	python >= 3.5
____________________________________________________________________________________
To train the model:
	1.First, set the arguments and hyperparameters in 'main.py' as follows:
		self.train = True
        	self.dev = True
        	self.test = False
		self.best_model_path = ''
		
	2.Then, run the following command in terminal
		python main.py
____________________________________________________________________________________
To Test the model:
	1.First, set the arguments and hyperparameters in 'main.py' as follows:
		self.train = False
        	self.dev = False
        	self.test = True
		self.best_model_path = self.save_path + '..._dev_best_model'  (where ... should be replaced with the biggest no of the saved models)
	2.Then, run the following command in terminal
		python main.py
____________________________________________________________________________________
