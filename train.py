import argparse
from process.utils import dataloader, train, test
from model.CNN import Conv_1d
from model.CNN_attention import *
import torch
import pickle



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='dataset/JNU/dataset.cpkl', help='load data')
parser.add_argument('--weight_path', type=str, default='model_weight', help='save or load best model.pth')
parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size')
parser.add_argument('--num_epoch', type=int, default=200, help='Number of epoch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--in_channel', type=int, default=1, help='in_channel')
parser.add_argument('--weight_decay', type=float, default=0.000005, help='Weight decay')
parser.add_argument('--n_class', type=int, default=4, help='Classes')
parser.add_argument('--reduction', type=int, default=2, help='Reduce channel')
parser.add_argument('--model_type', type=str, default='CNN', help='choice model')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(args.data_path, 'rb') as f:
    data, labels = pickle.load(f)
f.close()

train_loader, test_loader, val_loader = dataloader(data, labels)

if args.model_type == 'CNN':
    model = Conv_1d(args.in_channel, args.dropout, args.n_class)
elif args.model_type == 'SE_Conv':
    model = SE_Conv(args.in_channel, args.dropout, args.reduction, args.n_class)
elif args.model_type == 'CBAM_Conv':
    model = CBAM_Conv(args.in_channel, args.reduction)

train(train_loader, val_loader, model, args.num_epoch, args.lr, args.weight_decay, args.patience, device, args.weight_path)
test(model, args.weight_path, test_loader, device)



