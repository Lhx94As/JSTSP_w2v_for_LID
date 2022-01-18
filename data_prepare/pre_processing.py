import os
import json
import torch
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import s3prl.upstream.wav2vec2.hubconf as hubconf


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


def wav_lang_extract(wavscp, utt2lang):
    with open(wavscp, 'r') as f:
        lines_wav = f.readlines()
    audio_list = [x.split()[-1].strip() for x in lines_wav]
    with open(utt2lang,' r') as f:
        lines_utt = f.readlines()
    label_list = [x.split()[-1].strip().replace('-', '') for x in lines_utt]
    return audio_list, label_list

def feat_extract(wav2vec2lang, model, layer, save_dir, audio_list, label_list, device):
    with open(wav2vec2lang, 'w') as f:
        for i in tqdm(range(len(audio_list))):
            audio = audio_list[i]
            data, sr = librosa.load(audio, sr=None)
            data_ = torch.tensor(data).to(device=device, dtype=torch.float).unsqueeze(0)
            try:
                features = model(data_)
                features = features['hidden_state_{}'.format(layer)]
                features_ = features.squeeze(0).cpu().detach().numpy()
                new_feat = make_200ms_feat(features_, overlap=0, chunk_len=20)
                save_name = audio.replace(os.path.split(audio[0]), save_dir).replace('.wav', 'w2v_{}.npy'.format(layer))
                np.save(save_name, new_feat)
                f.write("{} {} {}\n".format(save_name, label_list[i], new_feat.shape[0]))
            except:
                print("Len:{} {} fail to extract".format(len(data) / sr, audio))


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--json', type=str, default='xsa_config.json')
    args = parser.parse_args()

    with open(args.json, 'r') as json_obj:
        config_proj= json.load(json_obj)

    le = LabelEncoder()
    device = torch.device('cuda:{}'.format(config_proj["optim_config"]["device"])
                          if torch.cuda.is_available() else 'cpu')
    model_path = config_proj["Input"]["userroot"] + config_proj["wav2vec_info"]["model_path"]
    model = hubconf.wav2vec2_local(ckpt=model_path)
    model.to(device)
    feat_layer = config_proj["wav2vec_info"]["layer"]
    wav_scp_train = config_proj["Input"]["userroot"] + config_proj["data_preprocess"]["wavscp_train"]
    utt2lang_train = config_proj["Input"]["userroot"] + config_proj["data_preprocess"]["utt2lang_train"]
    audio_train, labels_train = wav_lang_extract(wav_scp_train, utt2lang_train)
    labels_train_index = le.fit_transform(labels_train)
    save_w2v_train_dir = config_proj["Input"]["userroot"] + \
                         config_proj["data_preprocess"]["wavscp_train"].replace('/wav.scp', "/wav2vec_train/")
    if not os.path.exists(save_w2v_train_dir):
        os.mkdir(save_w2v_train_dir)
    train_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["train"]
    feat_extract(wav2vec2lang=train_txt, model=model, layer=feat_layer, save_dir=save_w2v_train_dir,
                 audio_list=audio_train, label_list=labels_train_index, device=device)

    if config_proj["data_preprocess"]["wavscp_valid"] != "none":
        wav_scp_valid = config_proj["Input"]["userroot"] + config_proj["data_preprocess"]["wavscp_valid"]
        utt2lang_valid = config_proj["Input"]["userroot"] + config_proj["data_preprocess"]["utt2lang_valid"]
        audio_valid, labels_valid = wav_lang_extract(wav_scp_valid, utt2lang_valid)
        labels_valid_index = le.transform(labels_valid)
        save_w2v_valid_dir = config_proj["Input"]["userroot"] + \
                             config_proj["data_preprocess"]["wavscp_valid"].replace('/wav.scp', "/wav2vec_valid/")
        if not os.path.exists(save_w2v_valid_dir):
            os.mkdir(save_w2v_valid_dir)
        valid_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["valid"]
        feat_extract(wav2vec2lang=valid_txt, model=model, layer=feat_layer, save_dir=save_w2v_valid_dir,
                     audio_list=audio_valid, label_list=labels_valid_index, device=device)
    if config_proj["data_preprocess"]["wavscp_test"] != "none":
        wav_scp_test = config_proj["Input"]["userroot"] + config_proj["data_preprocess"]["wavscp_test"]
        utt2lang_test = config_proj["Input"]["userroot"] + config_proj["data_preprocess"]["utt2lang_test"]
        audio_test, labels_test = wav_lang_extract(wav_scp_test, utt2lang_test)
        labels_test_index = le.transform(labels_test)
        save_w2v_test_dir = config_proj["Input"]["userroot"] + \
                             config_proj["data_preprocess"]["wavscp_test"].replace('/wav.scp', "/wav2vec_test/")
        if not os.path.exists(save_w2v_test_dir):
            os.mkdir(save_w2v_test_dir)
        test_txt = config_proj["Input"]["userroot"] + config_proj["Input"]["test"]
        feat_extract(wav2vec2lang=test_txt, model=model, layer=feat_layer, save_dir=save_w2v_test_dir,
                     audio_list=audio_test, label_list=labels_test_index, device=device)





if __name__ == "__main__":
    main()







