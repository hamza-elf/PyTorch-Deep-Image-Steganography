# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: main.py
@time: 2018/3/20

"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=False,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss

    #################  output configuration   ###############
    opt = parser.parse_args()
    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)
    opt.Hnet = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"
    opt.Rnet = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"
    testdir = opt.test
    test_dataset = MyImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize([opt.imageSize, opt.imageSize]),
            transforms.ToTensor(),
        ]))
    assert test_dataset

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.apply(weights_init)
    Hnet.load_state_dict(torch.load(opt.Hnet, map_location=map_location))
    print_network(Hnet)

    Rnet = RevealNet(output_function=nn.Sigmoid)
    Rnet.apply(weights_init)
    Rnet.load_state_dict(torch.load(opt.Rnet, map_location=map_location))
    print_network(Rnet)

    # MSE loss
    #     criterion = nn.MSELoss().cuda()
    criterion = nn.MSELoss()
    # test mode
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                             shuffle=False, num_workers=int(opt.workers))
    test(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
    print(
        "##################   test is completed  ######################")


def test(test_loader, epoch, Hnet, Rnet, criterion):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()  # record the Hloss in one epoch
    Rlosses = AverageMeter()  # record the Rloss in one epoch
    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step 

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchSize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # concat cover and original secret to get the concat_img with 6 channels
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        concat_imgv = Variable(concat_img, volatile=True)  # concat_img as input of Hiding net
        cover_imgv = Variable(cover_img, volatile=True)  # cover_imgv as label of Hiding net

        container_img = Hnet(concat_imgv)  # take concat_img as input of H-net and get the container_img
        errH = criterion(container_img, cover_imgv)  # H-net reconstructed error
        Hlosses.update(errH.data, this_batch_size)

        rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"
        secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
        errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error
        Rlosses.update(errR.data, this_batch_size)
        save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                        opt.testPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(this_batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    vutils.save_image(ContainerImg, '%s/Container.png' % opt.test)
    vutils.save_image(RevSecImg, '%s/RevSec.png' % opt.test)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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


if __name__ == '__main__':
    main()
