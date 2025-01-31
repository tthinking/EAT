import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import glob

from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Networks.Mymodules import MODEL as net
from Networks.Mymodules import DIS  as dis


from losses import CharbonnierLoss_loss, d_fake, d_real, gradient_penalty,ssim_loss


device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
else:
    print('CPU Mode Acitavted')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='...', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float)
    parser.add_argument('--weight', default=[1, 1,5], type=float)

    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    args = parser.parse_args()

    return args

class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):

        ir = '...'
        vi = '...'
        GT = '...'

        ir = Image.open(ir).convert('L')
        vi = Image.open(vi).convert('L')
        GT = Image.open(GT).convert('L')

        if self.transform is not None:
            tran = transforms.ToTensor()
            ir = tran(ir)

            vi = tran(vi)
            GT = tran(GT)
            input = torch.cat((ir, vi), -3)

            return ir, vi, GT,input

    def __len__(self):
        return len(self.imageFolderDataset)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

lambda_gp=10
def train(args, train_loader_ir, model,discri, criterion_CharbonnierLoss,criterion_ssimLoss, criterion_d_real, criterion_d_fake,criterion_gp, optimizerG,optimizerD):
    losses = AverageMeter()
    losses_CharbonnierLoss = AverageMeter()

    losses_ssimLoss = AverageMeter()
    losses_lossD = AverageMeter()
    losses_d_real = AverageMeter()
    losses_d_fake = AverageMeter()
    losses_d_gp = AverageMeter()

    weight = args.weight
    model.train()
    discri.train()

    for i, (ir,vi,GT, input)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):

        if use_gpu:
            input = input.cuda()

            GT = GT.cuda()

        else:
            input = input

            GT=GT

        out = model(input)
        d_fake=discri(out)

        d_real =discri(GT)


        alpha = torch.rand(GT.size(0), 1, 1, 1).cuda()
        img_hat = (alpha * GT.data + (1 - alpha) * out.data).requires_grad_(True)
        d_hat_decision =discri(img_hat)


        loss_CharbonnierLoss = weight[0] * criterion_CharbonnierLoss(out,GT)

        loss_ssimLoss= weight[1] * criterion_ssimLoss(out,GT)

        loss_d_fake = criterion_d_fake(d_fake)

        loss = loss_CharbonnierLoss + loss_ssimLoss - 1 *loss_d_fake

        losses.update(loss.item(), input.size(0))
        losses_CharbonnierLoss.update(loss_CharbonnierLoss.item(), input.size(0))

        losses_ssimLoss.update(loss_ssimLoss.item(), input.size(0))

        d_fake = discri(out.detach())
        loss_d_fake = criterion_d_fake(d_fake)
        loss_d_real = - criterion_d_real(d_real)
        loss_d_gp = criterion_gp(d_hat_decision, img_hat)

        lossD = loss_d_real+ loss_d_fake+ lambda_gp* loss_d_gp


        losses_lossD.update(lossD.item(), input.size(0))
        losses_d_real.update(loss_d_real.item(), input.size(0))
        losses_d_fake.update(loss_d_fake.item(), input.size(0))
        losses_d_gp.update(loss_d_gp.item(), input.size(0))


        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()


        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_CharbonnierLoss', losses_CharbonnierLoss.avg),
        ('loss_ssimLoss', losses_ssimLoss.avg),
        ('lossD', losses_lossD.avg),
        ('loss_d_real', losses_d_real.avg),
        ('loss_d_fake',losses_d_fake.avg),
        ('loss_d_gp',losses_d_gp.avg),

    ])

    return log



def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    training_dir_ir = "..."
    folder_dataset_train_ir = glob.glob(training_dir_ir )
    training_dir_vi = "..."

    folder_dataset_train_vi= glob.glob(training_dir_vi )
    training_dir_GT = ".../"

    folder_dataset_train_GT = glob.glob(training_dir_GT )

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])

    dataset_train_ir = GetDataset(imageFolderDataset=folder_dataset_train_ir,
                                                  transform=transform_train)
    dataset_train_vi = GetDataset(imageFolderDataset=folder_dataset_train_vi,
                                  transform=transform_train)
    dataset_train_GT = GetDataset(imageFolderDataset=folder_dataset_train_GT,
                                  transform=transform_train)


    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.batch_size)
    train_loader_vi = DataLoader(dataset_train_vi,
                                 shuffle=True,
                                 batch_size=args.batch_size)
    train_loader_GT = DataLoader(dataset_train_GT,
                                 shuffle=True,
                                 batch_size=args.batch_size)
    model = net(in_channel=2)
    discri = dis(in_channel=1)
    if use_gpu:
        model = model.cuda()
        model.cuda()
        discri =  discri.cuda()
        discri.cuda()

    else:
        model = model
        discri=discri
    criterion_CharbonnierLoss = CharbonnierLoss_loss

    criterion_ssimLoss = ssim_loss
    criterion_d_fake = d_fake
    criterion_d_real = d_real
    criterion_d_gp =  gradient_penalty
    optimizerG = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    optimizerD = optim.Adam(discri.parameters(), lr=args.lr,
                            betas=args.betas, eps=args.eps)

    log = pd.DataFrame(index=[],
                       columns=['epoch',

                                'loss',
                                'loss_CharbonnierLoss',
                                'loss_ssimLoss',
                                'loss_d_real',
                                'loss_d_fake',
                                'loss_d_gp',
                                'lossD' ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader_ir,train_loader_vi,  train_loader_GT, model,discri, criterion_CharbonnierLoss,criterion_ssimLoss, criterion_d_real,criterion_d_fake,  criterion_d_gp,  optimizerG, optimizerD, epoch)     # 训练集


        print('loss: %.4f - loss_CharbonnierLoss: %.4f- loss_ssimLoss: %.4f - lossD: %.4f -  loss_d_real:  %.4f -  loss_d_fake:  %.4f - loss_d_gp:  %.4f '
              % (train_log['loss'],
                 train_log['loss_CharbonnierLoss'],
                 train_log['loss_ssimLoss'],
                 train_log['lossD'],
                 train_log['loss_d_real'],
                 train_log['loss_d_fake'],
                 train_log['loss_d_gp'] ))



        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['loss_CharbonnierLoss'],
            train_log['loss_ssimLoss'],
            train_log['lossD'],
            train_log['loss_d_real'],
            train_log['loss_d_fake'],
            train_log['loss_d_gp'],
        ], index=['epoch', 'loss', 'loss_CharbonnierLoss', 'loss_ssimLoss',  'lossD',  'loss_d_real',  'loss_d_fake',  'loss_d_gp'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)
            torch.save(discri.state_dict(), 'models/%s/discri_{}.pth'.format(epoch + 1) % args.name)


if __name__ == '__main__':
    main()


