# JSTSP
Before running the training scripts, pls download asv-subtools and move the folder "subtool" to your kaldi/esg/  
Pls also download s3prl and run the data_preprocessing.py in s3prl.

E.g to run these files  
data_preprocessing:  
>python data_prepocessing.py --step 0 --lredir /home/user/LDC_LRE2017_Training_Data/ --savedir /home/user/your_save_dir/ --segment /home/user/segment --model /home/xlsr.pt --kaldi /home/user/kaldi/ --layer 16 --seglen 30 --overlap 1 --filerange 0_10000 --device 0

Then run combine_fea2lang.py if we have run multiple jobs. The output is /home/user/your_save_dir/feat2lang_all.txt, each line feat:  
>python combine_feat2lang.py --datadir /home/user/your_save_dir/

train_xsa.py:  
>python train_xsa.py --model model_Tom --kaldi /home/user/kaldi/ ---train /home/wav2vec_features_train.txt --valid /home/wav2vec_features_test.txt --batch 128 --epochs 20 --lang 14 --device 0
