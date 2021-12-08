# JSTSP
Before running the training scripts, pls download asv-subtools and move the folder "subtool" to your kaldi/esg/  
Pls also download s3prl and run the data_preprocessing.py in s3prl.

E.g to run these files  
data_preprocessing:  
>python data_prepocessing.py --step 0 --lredir /home/user/LDC_LRE2017_Training_Data/ --savedir /home/user/save_dir/ --segment /home/user/segment--model /home/xlsr.pt --kaldi /home/user/kaldi/ --layer 16 --seglen 30 --overlap 1 --device 0
  
train_xsa.py:  
>python train_xsa.py --model model_Tom --kaldi /home/user/kaldi/ ---train /home/wav2vec_features.txt --batch 128 --epochs 20 --lang 14 --device 0
