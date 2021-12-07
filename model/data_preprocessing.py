import os
import torch
import librosa
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import s3prl.upstream.wav2vec2.hubconf as hubconf
from sklearn.preprocessing import LabelEncoder


def make_200ms_feat(mfccs, overlap=10, chunk_len=20):
    new_feat = 0
    feature = mfccs
    seq_len = feature.shape[0]
    step = chunk_len - overlap
    num_chunk = (seq_len - overlap) // (chunk_len - overlap)
    if num_chunk > 1:
        start = 0
        end = 0
        for id in range(num_chunk):
            end = start + chunk_len
            feat_temp = feature[start:end, :]
            feat_temp = np.hstack(feat_temp)
            start += step
            if id == 0:
                new_feat = feat_temp
            else:
                new_feat = np.vstack((new_feat, feat_temp))
        num_left = seq_len - end
        start = end - (chunk_len - num_left)
        feat_temp = feature[start:, :]
        feat_temp = np.hstack(feat_temp)
        new_feat = np.vstack((new_feat, feat_temp))
    return new_feat

def upsampling_lre(audio, save_dir):
    if audio.endswith('.sph'):
        new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.sph', '.wav')
        subprocess.call(f"sph2pipe -p {audio} {new_name}", shell=True)
        subprocess.call(f"sox {new_name} -r 16000 {new_name}", shell=True)
    elif audio.endswith('.wav') or audio.endswith('.WAV'):
        new_name = save_dir + '/' + os.path.split(audio)[-1].replace('.WAV', '.wav')
        subprocess.call(f"sox {audio} -r 16000 {new_name}", shell=True)

def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--step', type=int, help='step to control the process. 0 means starting from beginning',
                        default=0)
    parser.add_argument('--lredir', type=str, help='e.g.: lre_train/')
    parser.add_argument('--model', type=str, help='pretrained XLSR-53 model dir')
    parser.add_argument('--kaldi', type=str, help='e.g.: /home/user_kk/kaldi/')
    parser.add_argument('--device',type = int, help='cuda id, default 0', default=0)
    parser.add_argument('--layer', type=int, help='Extract wav2vec feats from this layer', default=16)
    parser.add_argument('--seglen', type=int, help='segmentlength', default=30)
    parser.add_argument('--overlap', type=int, help='overlap length', default=1)
    parser.add_argument('--savedir', type=str, help='dir to save wav2vec feats', default=None)
    args = parser.parse_args()
    os.environ["PATH"] += os.pathsep + "{}/tools/sph2pipe_v2.5/".format(args.kaldi)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    model = hubconf.wav2vec2_local(ckpt=args.model)
    model.to(device)
    dir_list = ['araacm', 'araapc', 'araary', 'araarz', 'enggbr', 'engusg', 'porbrz', 'qslpol',
                'qslrus', 'spacar', 'spaeur', 'spalac', 'zhocmn', 'zhonan']
    le = LabelEncoder()
    le.fit(dir_list)
    save_dir = args.savedir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.step <= 0:
        print('The upsampled data will be stored in {}'.format(save_dir))
        audio_list = []
        labels = []
        lredir = args.lredir
        if lredir.endswith('Training_Data/'):
            key_file = args.lredir + '/docs/filename_language_key.tab'
            with open(key_file, 'r') as f:
                lines = f.readlines()
            labels_text = [x.split()[1].strip().replace('-', '') for x in lines]
            audio_list = [args.lredir + '/data/{}/{}'.format(x.split()[1].strip(),x.split()[0]) for x in lines]
            labels = le.transform(labels_text)
        elif lredir.endswith('Development_Data/'):
            key_file = args.lredir + '/docs/lre17_dev_segments.key'
            with open(key_file, 'r') as f:
                lines = f.readlines()
            labels_text = [x.split()[1].strip().replace('-', '') for x in lines[1:]]
            audio_list = [args.lredir + '/data/dev/{}'.format(x.split()[0]) for x in lines[1:]]
            labels = le.transform(labels_text)
        elif lredir.endswith('Eval_Data/'):
            key_file = args.lredir + '/docs/lre17_eval_segments.key'
            with open(key_file, 'r') as f:
                lines = f.readlines()
            labels_text = [x.split()[1].strip().replace('-', '') for x in lines[1:]]
            audio_list = [args.lredir + '/data/{}'.format(x.split()[0]) for x in lines[1:]]
            labels = le.transform(labels_text)


        audio2lang_txt = save_dir + '/wav2lang.txt'
        with open(audio2lang_txt, 'w') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                try:
                    upsampling_lre(audio, save_dir)
                    save_name = save_dir+'/'+os.path.split(audio)[-1].replace('.WAV','.wav').replace('.sph','wav')
                    f.write("{} {}\n".format(save_name, labels[i]))
                except:
                    print(audio)
    if args.step <= 1:
        print('Completed upsampling, segmenting long utterances to {} secs'.format(args.seglen))
        
        audio2lang_txt = save_dir + '/wav2lang.txt'
        with open(audio2lang_txt, 'r') as f:
            lines = f.readlines()
        audio_list = [x.split()[0] for x in lines]
        labels_list = [x.split()[1].strip() for x in lines]
        audio2lang_seg_txt = save_dir + '/segment2lang.txt'
        seg_len = args.seglen
        overlap = args.overlap
        save_seg_dir = args.savedir + '/segs/'
        print('The segmented data will be stored in {}'.format(save_seg_dir))
        if not os.path.exists(save_seg_dir):
            os.mkdir(save_seg_dir)
        with open(audio2lang_seg_txt, 'w') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                main_name = os.path.split(audio)[-1]
                label = labels_list[i]
                try:
                    audio_length = librosa.get_duration(filename=audio)
                    data_ = AudioSegment.from_file(audio, "wav")
                    num_segs = (audio_length - overlap) // (seg_len - overlap)
                    remainder = (audio_length - overlap) % (seg_len - overlap)
                    if num_segs >= 1:
                        start = 0
                        for ii in range(num_segs):
                            end = start + seg_len
                            start_ = start * 1000
                            end_ = end * 1000
                            data_seg = data_[start_:end_]
                            save_name = save_seg_dir + main_name.replace('.wav', '_{}.wav'.format(ii))
                            data_seg.export(save_name, format='wav')
                            start = end - overlap
                            f.write("{} {}\n".format(save_name, label))
                        if remainder >= 3: # if remainder longer than 3s, keep it, otherwise throw away
                            start_ = start * 1000
                            data_seg = data_[start_:]
                            save_name = save_seg_dir + main_name.replace('.wav', '_{}.wav'.format(num_segs))
                            data_seg.export(save_name, format='wav')
                            f.write("{} {}\n".format(save_name, label))
                except:
                    print('Errors when segmenting')
    if args.step <= 2:
        save_seg_dir = args.savedir + '/segs/'
        print("Extracting wav2vec features from layer {} of pretrained {}".
              format(args.layer, os.path.split(args.model)[-1].split('.')[0]))
        print('The features will be stored in {}'.format(save_seg_dir))
        audio2lang_seg_txt = save_dir + '/segment2lang.txt'
        feat2lang_txt = save_dir + '/feat2lang.txt'
        with open(audio2lang_seg_txt, 'r') as f:
            lines = f.readlines()
        audio_list = [x.split()[0] for x in lines]
        labels_list = [x.split()[1].strip() for x in lines]
        with open(feat2lang_txt, 'w') as f:
            for i in tqdm(range(len(audio_list))):
                audio = audio_list[i]
                label = labels_list[i]
                data, sr = librosa.load(audio, sr=None)
                data_ = torch.tensor(data).to(device=device, dtype=torch.float).unsqueeze(0)
                try: # To skip some too long or too short utterances
                    features = model(data_)
                    features = features['hidden_state_{}'.format(args.layer)]
                    features_ = features.squeeze(0).cpu().detach().numpy()
                    save_name = audio.replace('.wav', '.npy')
                    new_feat = make_200ms_feat(features_, overlap=0, chunk_len=20)
                    np.save(save_name, new_feat)
                    f.write("{} {} {}\n".format(save_name, label, features.shape[0]))
                except:
                    print("Len:{} {} is not successful, skip this one".format(len(data) / sr, audio))

if __name__ == "__main__":
    main()
