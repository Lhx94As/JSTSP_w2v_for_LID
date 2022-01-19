# JSTSP
Before running the training scripts, 
1. Download asv-subtools and move the folder "subtool" to your kaldi/esg/  
2. Download s3prl and run the pre_processing.py in s3prl to extract the features.

An example:
>pre_processing.py --json xsa_config.json  
>python train_xsa.py --json xsa_config.json

Before running, pls revise the json configuration file according to your own root.  
Mainly the "Input" part, you can keep others since they are the parameters I am using :)

