import argparse
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import div2k,benchmark
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
import skimage.color
import skimage.measure
from Model2 import DenseNet
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--n_threads', type=int, default=3,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='./dataset/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K2',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K2',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=5,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep_reset',
                    help='dataset file extension')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=120,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')


parser.add_argument('--num_epochs', default=500, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

train_set = div2k.DIV2K(args = opt,train=True)
val_set = ValDatasetFromFolder('./Set5', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = DenseNet()
netG = torch.nn.DataParallel(netG,device_ids=[0,1])

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))


generator_criterion = torch.nn.MSELoss()

if torch.cuda.is_available():
    netG.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters(),lr=1e-4)
optimizerG = torch.nn.DataParallel(optimizerG,device_ids=[0,1]).module

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
Maxapsnr = 0
scheduler = lr_scheduler.StepLR(optimizerG,step_size=200,gamma=0.5)
ss = 0
for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    scheduler.step()
    netG.train()
    for data, target in train_bar:
        data = data/255.0
        target = target/255.0
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)

        netG.zero_grad()
        g_loss = generator_criterion(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        train_bar.set_description(desc='%f' % (g_loss.data[0]))
        running_results['g_loss']+=g_loss.data[0]


    ss +=1
    netG.eval()
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    result = 0
    result1 = 0
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = Variable(val_lr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        srdata = sr.data[0].permute(2, 1, 0).cpu().numpy()
        hrdata = hr.data[0].permute(2, 1, 0).cpu().numpy()

        srdata[srdata < 0.0] = 0.0
        srdata[srdata > 1.0] = 1.0
        srdata = skimage.color.rgb2ycbcr(srdata).astype('uint8')
        hrdata = skimage.color.rgb2ycbcr(hrdata).astype('uint8')
        bb = skimage.measure.compare_psnr(hrdata[:, :, 0], srdata[:, :, 0])
        cc = skimage.measure.compare_ssim(hrdata[:, :, 0], srdata[:, :, 0])
        result += bb
        result1 += cc

    valing_results['psnr'] = result/valing_results['batch_sizes']
    valing_results['ssim'] = result1/valing_results['batch_sizes']
    # save model parameters
    if valing_results['psnr'] > Maxapsnr:
        torch.save(netG.state_dict(), 'epochs/SRWDNDIV2k.pth')
        Maxapsnr = valing_results['psnr']
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])

    out_path = 'statistics/'
    data_frame = pd.DataFrame(
        data={'g_loss': results['g_loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        index=range(1, ss + 1))
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_MYDenseDIV2k.csv', index_label='Epoch')
