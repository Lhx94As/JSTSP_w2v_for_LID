import torch

from model import *
from data_load import *
import scoring
import subprocess


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--dim', type=int, help='dim of input features',  default=1024)
    parser.add_argument('--featdim', type=int, help='dim of input of attention blocks', default=64)
    parser.add_argument('--head', type=int, help='num of attention heads', default=8)
    parser.add_argument('--model', type=str, help='model name',  default='AE_XSA')
    parser.add_argument('--kaldi', type=str, help='kaldi root', default='/home/hexin/Desktop/kaldi')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    parser.add_argument('--batch', type=int, help='batch size', default=128)
    parser.add_argument('--warmup', type=int, help='num of epochs')
    parser.add_argument('--epochs', type=int, help='num of epochs', default=10)
    parser.add_argument('--lang', type=int, help='num of language classes', default=10)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--device', type=int, help='Device name', default=0)
    parser.add_argument('--seed', type=int, help='seed, for XSA setting seed slow the training', default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')


    model = X_Transformer_E2E_LID(input_dim=args.dim,
                                  feat_dim=args.featdim,
                                  d_k=args.featdim,
                                  d_v=args.featdim,
                                  d_ff=2048,
                                  n_heads=args.head,
                                  dropout=0.1,
                                  n_lang=args.lang,
                                  max_seq_len=10000)

    model.to(device)

    train_txt = args.train
    train_set = RawFeatures(train_txt)
    valid_txt = '/home/hexin/Desktop/hexin/dataset/OLR2020/AP19-OLR_data/wav2vec2lang_16_test.txt'
    valid_set = RawFeatures(valid_txt)
    train_data = DataLoader(dataset=train_set,
                            batch_size=args.batch,
                            pin_memory=True,
                            num_workers=8,
                            shuffle=True,
                            collate_fn=collate_fn_atten)
    valid_data= DataLoader(dataset=valid_set,
                           batch_size=1,
                           pin_memory=True,
                           shuffle=False,
                           collate_fn=collate_fn_atten)

    loss_func_CRE = nn.CrossEntropyLoss().to(device)
    total_step = len(train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if not args.warmup:
        warmup = total_step*3
    else:
        warmup = args.warnup
    warm_up_with_cosine_lr = lambda step: step / warmup \
        if step <= warmup \
        else 0.5 * (math.cos((step - warmup) / (args.epochs * total_step - warmup) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)



    for epoch in tqdm(range(args.epochs)):
        model.train()
        for step, (utt, labels, seq_len) in enumerate(train_data):
            utt_ = utt.to(device=device)
            seq_weights = seq_len[0] / torch.tensor(seq_len).to(device=device)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            std_mask_, weight_unbaised = std_mask(seq_len, len(seq_len), dim=args.featdim * args.head)
            std_mask_ = std_mask_.to(device=device)
            weight_unbaised = weight_unbaised.to(device=device)
            labels = labels.to(device=device)
            # Forward pass
            outputs = model(utt_, seq_len, seq_weights, std_mask_, weight_unbaised, atten_mask=atten_mask)
            loss_lid = loss_func_CRE(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss_lid.backward()
            optimizer.step()
            scheduler.step()
            if step % 200 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} ".
                      format(epoch + 1, args.epochs, step + 1, total_step, loss_lid.item()))


            # print(get_lr(optimizer))

        if epoch >= args.epochs - 10:
            torch.save(model.state_dict(), '{}_epoch_{}.ckpt'.format(args.model, epoch))
            model.eval()
            correct = 0
            total = 0
            scores = 0
            with torch.no_grad():
                for step, (utt, labels, seq_len) in enumerate(valid_data):
                    utt = utt.to(device=device, dtype=torch.float)
                    seq_weights = seq_len[0] / torch.tensor(seq_len).to(device=device)
                    std_mask_, weight_unbaised = std_mask(seq_len, len(seq_len), dim=args.featdim * args.head)
                    std_mask_ = std_mask_.to(device=device)
                    weight_unbaised = weight_unbaised.to(device=device)
                    labels = labels.to(device)
                    # Forward pass\
                    outputs = model(utt, seq_len, seq_weights, std_mask_, weight_unbaised, atten_mask=None)
                    predicted = torch.argmax(outputs, -1)
                    total += labels.size(-1)
                    correct += (predicted == labels).sum().item()
                    if step == 0:
                        scores = outputs
                    else:
                        scores = torch.cat((scores, outputs), dim=0)
            acc = correct / total
            print('Current Acc.: {:.4f} %'.format(100 * acc))
            scores = scores.squeeze().cpu().numpy()
            print(scores.shape)
            trial_txt = os.path.split(args.train)[0] + '/trial_{}.txt'.format(args.model)
            score_txt = os.path.split(args.train)[0] + '/score_{}.txt'.format(args.model)
            scoring.get_trials(valid_txt, args.lang, trial_txt)
            scoring.get_score(valid_txt, scores, args.lang, score_txt)
            eer_txt = trial_txt.replace('trial', 'eer')
            subprocess.call(f"{args.kaldi}/egs/subtools/computeEER.sh "
                            f"--write-file {eer_txt} {trial_txt} {score_txt}", shell=True)
            cavg = scoring.compute_cavg(trial_txt, score_txt)
            print("Cavg:{}".format(cavg))


if __name__ == "__main__":
    main()
