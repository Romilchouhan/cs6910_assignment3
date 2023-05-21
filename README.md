Code repository for the Assignment 3 of the course CS6910: Fundamentals of Deep Learning offered at IIT Madras during Jan-May 2021. <br>
The assignment is about creating a Seq2Seq model for character-to-character word transliteration task on Dakshina Dataset. <br>

I have created separate python files for building dataset: `dataset.py`, building model: `network1.py`, building attention model: `attention.py` and training model: `train.py`. <br>

`train.py` is the main file which imports the other files and trains the model. It has several command line arguments of which some are mandatory and some are optional. The optional ones are the default hyperparameters values which can be changed by the user. The two mandatory arguments are: <br>
1. '--wandb': This argument is used to enable wandb sweeps and logging. It is a boolean argument and can be set to either True or False. <br>
2. '--test': This argument is used to enable testing. It is a boolean argument and can be set to either True or False. If this argument i true, the model will get automatically run with the best hyperparameter found from the wandb sweeps. It will generate accuracy with and without attention models. It also generates prediction of the non-attention model on randomly selected batches from test data. <br>

If none of the mandatory arguments are given, the model will run with default hyperparameter values and generate train and validation accuracy. <br>

The wandb report for the following assignment can be found at: [https://wandb.ai/romilchouhan/A3%20trial%20tamil%20count%20120/reports/EE20B112-Assignment-3-Report--Vmlldzo0NDE3Mjc4](Report)