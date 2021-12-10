import glob
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--datadir', type=str,
                        help='where you save feat2lang files, same as the savedir in w2v_feat_extraction.py')
    args = parser.parse_args()
    data_dir = args.datadir
    txt_file = glob.glob(data_dir+'/*txt')
    feat2lang_files = [x for x in txt_file if os.path.split(x)[-1].startswith("feat2lang_")]
    feat2lang_20frams = data_dir + '/feat2lang_all.txt'
    with open(feat2lang_20frams, 'w') as f:
        for feat2lang_ in feat2lang_files:
            with open(feat2lang_, 'r') as ff:
                lines = ff.readlines()
            for line_ in lines:
                f.write(line_)
