from data_utils import TestDatasetFromFolder
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import skimage.io
import Model2

parser = argparse.ArgumentParser(description='Test Single Image')

# Setting
parser.add_argument('--model_name', default='model.pkl', type=str)
parser.add_argument('--inputLRpath', default='./Testdata', type=str)
parser.add_argument('--outputdir', default='./outdir', type=str)

opt = parser.parse_args()


test_set = TestDatasetFromFolder(opt.inputLRpath, upscale_factor=4)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

# load model (double GPUs)
ADRD = torch.load(opt.model_name)

for i,(val_lr, bicubicimg) in enumerate(test_loader):

    lr = Variable(val_lr, volatile=True)

    if torch.cuda.is_available():
        lr = lr.cuda()

    #prediction
    sr = ADRD(lr)

    srdata = sr.data[0].permute(1, 2, 0).cpu().numpy()
    srdata[srdata < 0.0] = 0.0
    srdata[srdata > 1.0] = 1.0

    bicubicimg = bicubicimg.data[0].permute(1, 2, 0).cpu().numpy()

    # save image
    skimage.io.imsave(opt.outputdir + '/%d.png' % i, srdata)
    skimage.io.imsave(opt.outputdir + '/%d_bicubic.png' % i, bicubicimg)

