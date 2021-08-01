from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier2 as classifier
import classifier_entropy
import model
import numpy as np
import sys
import time
import pickle
import copy

parser = argparse.ArgumentParser("GZSL Action")
parser.add_argument('--save_name', default='baseline', help='saved directory name')
parser.add_argument('--dataset', default='vg', help='Dataset name')
parser.add_argument('--dataroot', default='data_action/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--action_embedding', default='i3d')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--gzsl_od', action='store_true', default=False, help='enable out-of-distribution based generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=4096, help='size of visual features')
parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cosem_weight', type=float, default=0.1, help='weight of the cos embed loss')
parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
parser.add_argument('--cls_weight', type=float, default=1, help='')
parser.add_argument('--cls_weight1', type=float, default=0.1, help='recons_weight for decoder')


parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--no_bg_file', type=int, default=26)
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_iou', default='', help="path to netG (to continue training)")
parser.add_argument('--netD_iou', default='', help="path to netD (to continue training)")
parser.add_argument('--netDec', default='', help="path to netDec (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--iou_information_size', type=int, default=256)
parser.add_argument('--file_number', type=int, default=4)
parser.add_argument('--folder_number', type=int, default=5)
parser.add_argument('--second_gan', type=int, default=5)
parser.add_argument('--all_gt_file', type=int, default=12)
parser.add_argument('--neg_times', dest='neg_times', help='neg_times', default=3, type=int)
parser.add_argument('--save_folder', dest='save_folder', help='save_folder', default='baseline', type=str)
parser.add_argument('--bg_generate', dest='bg_generate', help='bg_generate', default='seen', type=str)
parser.add_argument('--sv', dest='sv', help='sv', default='coco', type=str)

parser.add_argument('--loss_rec', dest='loss_rec', help='loss_rec', default=False, type=bool)

parser.add_argument('--total_file_number', dest='total_file_number', help='total_file_number', default=3, type=int)

parser.add_argument('--test_code', dest='test_code', help='test_code', default='no', type=str)

parser.add_argument('--loss_mum', dest='loss_mum', help='loss_mum', default=False, type=bool)

parser.add_argument('--loss_cls', dest='loss_cls', help='loss_cls', default=False, type=bool)


def emb_criterion(x1, x2, label, attri=[], attri_um=[]):

    cos_son = torch.sum(x1 * x2, 1)
    cos_mother1 = torch.sum(x1 ** 2, 1) ** 0.5
    cos_mother2 = torch.sum(x2 ** 2, 1) ** 0.5
    cos_mother = cos_mother1 * cos_mother2

    cos_similarity = cos_son / cos_mother

    part1 = 1 - cos_similarity[label==1]

    index_part2 = label==-1
    part2 = cos_similarity[index_part2]

    # part2 = part2[part2 > 0]


    if torch.sum(index_part2) > 0:
        attri_part2 = attri[index_part2, :]
        attri_um_part2 = attri_um[index_part2, :]

        weight = torch.sum((attri_part2 - attri_um_part2) ** 2, 1)

        part2 = weight * part2

        part2[part2<0] = 0

        # attri_part2 = attri_part2[part2 > 0, :]
        # attri_um_part2 = attri_um_part2[part2 > 0, :]

        # part2 = weight * part2

    cos_similarity = torch.cat((part1, part2), 0)

    # cos_similarity = 1 - (cos_son / cos_mother)
    cos_similarity = torch.mean(cos_similarity)
    return cos_similarity




opt = parser.parse_args()
print(opt)
print(opt.dataset)
print(opt.bg_generate)

if opt.loss_rec:
    print('loss_rec')


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.Data_Loader(opt)
print("# of training samples: ", data.ntrain)

# initialize generator, discriminator and decoder
netD = model.MLP_CRITIC(opt)
netG = model.MLP_G(opt)
netDec = model.Dec(opt)

if opt.loss_cls:
    netCls1 = model.cls_loss_layer(opt, data.seenclasses.shape[0]-1)

"added"
netG_iou = model.MLP_G_IOU(opt)
# netG_iou = model.MLP_G_2nd(opt)
netD_iou = model.MLP_CRITIC(opt)

if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
    # netG_iou_bg = model.MLP_G_2nd(opt)
    netG_iou_bg = model.MLP_G_IOU(opt)
    netD_iou_bg = model.MLP_CRITIC(opt)
    if opt.loss_cls:
        netCls2 = model.cls_loss_layer(opt, data.seenclasses.shape[0])

# Load nets if paths present
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netDec != '':
    netDec.load_state_dict(torch.load(opt.netDec))

    "added"
if opt.netG_iou != '':
    netG_iou.load_state_dict(torch.load(opt.netG_iou))
if opt.netD_iou != '':
    netD_iou.load_state_dict(torch.load(opt.netD_iou))

# print nets
print("First Network Part")
print(netG)
print(netD)
print(netDec)

print("Second Network Part")
print(netG_iou)
print(netD_iou)

# emb_criterion = nn.CosineEmbeddingLoss(margin=0)
recons_criterion = nn.MSELoss()
cls_criterion = nn.CrossEntropyLoss()

# recons_criterion = nn.L1Loss()  # L1 loss 

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

"added"
input_res_iou = torch.FloatTensor(opt.batch_size, opt.resSize)
iou_inf = torch.FloatTensor(opt.batch_size, opt.iou_information_size)

input_res_iou_bg = torch.FloatTensor(opt.batch_size, opt.resSize)
iou_inf_bg = torch.FloatTensor(opt.batch_size, opt.iou_information_size)

noise_iou = torch.FloatTensor(opt.batch_size, opt.nz)
noise_iou_bg = torch.FloatTensor(opt.batch_size, opt.nz)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netDec.cuda()

    "added"
    netG_iou.cuda()
    netD_iou.cuda()
    if opt.loss_cls:
        netCls1.cuda()

    if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
        netG_iou_bg.cuda()
        netD_iou_bg.cuda()
        if opt.loss_cls:
            netCls2.cuda()

    input_res, input_label = input_res.cuda(), input_label.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()

    "added"
    input_res_iou, iou_inf = input_res_iou.cuda(), iou_inf.cuda()
    input_res_iou_bg, iou_inf_bg = input_res_iou_bg.cuda(), iou_inf_bg.cuda()

    noise_iou = noise_iou.cuda()
    noise_iou_bg = noise_iou_bg.cuda()


    one = one.cuda()
    mone = mone.cuda()
    # emb_criterion.cuda()
    recons_criterion.cuda()
    cls_criterion.cuda()
        
    
def sample():
    # Sample a batch

    "added"
    batch_feature, batch_label, batch_att, batch_feature_iou, batch_iou_inf, batch_feature_iou_bg, batch_iou_inf_bg \
        = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

    "added"
    input_res_iou.copy_(batch_feature_iou)
    iou_inf.copy_(batch_iou_inf.repeat(1, opt.iou_information_size))

    input_res_iou_bg.copy_(batch_feature_iou_bg)
    iou_inf_bg.copy_(batch_iou_inf_bg.repeat(1, opt.iou_information_size))

def generate_syn_feature(netG, classes, attribute, num):
    # generate num synthetic samples for each class in classes
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.ShortTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)

    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i].long()
        iclass_att = attribute[iclass, :].reshape(1, -1)
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass.item())
    return syn_feature, syn_label

def generate_syn_feature_iou(netG, classes, attribute, num, netG_iou, pos_iou, neg_iou, neg_times=3, bg_generate='seen', netG_iou_bg=None, dataset='coco'):
    # generate num synthetic samples for each class in classes
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.ShortTensor(nclass*num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    iou_inf = torch.FloatTensor(num, opt.iou_information_size)

    syn_feature_one_batch = torch.FloatTensor(num, opt.resSize)

    syn_feature_bg = torch.FloatTensor(nclass * num * neg_times, opt.resSize)
    syn_label_bg = torch.ShortTensor(nclass * num * neg_times)
    syn_att_bg = torch.FloatTensor(num * neg_times, opt.attSize)
    syn_noise_bg = torch.FloatTensor(num * neg_times, opt.nz)
    iou_inf_bg = torch.FloatTensor(num * neg_times, opt.iou_information_size)

    syn_feature_one_batch_bg = torch.FloatTensor(num * neg_times, opt.resSize)


    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        iou_inf = iou_inf.cuda()
        # syn_feature_one_batch = syn_feature_one_batch.cuda()

        syn_noise_bg = syn_noise_bg.cuda()
        iou_inf_bg = iou_inf_bg.cuda()
        # syn_feature_one_batch_bg = syn_feature_one_batch_bg.cuda()

    # pos_iou_save = copy.deepcopy(pos_iou)
    # neg_iou_save = copy.deepcopy(neg_iou)

    if bg_generate == 'seen':
        print('load seen bg feature' + dataset)
        bg_feature_list = []
        if dataset == 'coco':
            head = '/root/Workspace/zsz/gan_training_file/for_bg/for_bbg/bg_feature'
            file_num = 5
        elif dataset == 'vg':
            head = '/root/Workspace/zsz/gan_training_file_vg/train_feature/bg_feature/bg_feature2/bg_feature'
            file_num = 9
        elif dataset == 'imagenet':
            head = '/root/Workspace/zsz/gan_training_file_imagenet/training_feature/bg_feature/bg_feature0/bg_feature'
            file_num = 44
        bg_num = 0
        for bg_file_number in range(file_num):
            filename = head + str(bg_file_number) + '.pkl'
            with open(filename, 'rb') as f:
                bg_feature = pickle.load(f)
                bg_num += bg_feature.shape[0]
            bg_feature_list.append(bg_feature)
            if bg_num > nclass * num * neg_times:
                break
        bg_feature = np.concatenate(bg_feature_list, axis=0)
        bg_feature = bg_feature[:, 3:]



    for i in range(nclass):
        iclass = classes[i].long()
        iclass_att = attribute[iclass, :].reshape(1, -1)
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise), Variable(syn_att))
        # netG_iou(noisev_iou, fake, iouv_inf)

        syn_feature_one_batch.copy_(output.data.cpu())

        if bg_generate == 'syn_gt':
            syn_feature_one_batch_bg.copy_(output.data.cpu().repeat(neg_times, 1))

        # iou_inf.copy_(pos_iou[:num])
        # pos_iou = pos_iou[num:]
        #
        # if pos_iou.shape[0] < num:
        #     pos_iou = copy.deepcopy(pos_iou_save)

        syn_noise.normal_(0, 1)
        output = netG_iou(Variable(syn_noise), Variable(syn_feature_one_batch.cuda()))
        syn_label.narrow(0, i*num, num).fill_(iclass.item())
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())

        if bg_generate == 'seen':
            output_bg_feature = bg_feature[:neg_times * num, :]
            bg_feature = bg_feature[neg_times * num:, :]
            syn_feature_bg.narrow(0, i * num * neg_times, num * neg_times).copy_(torch.from_numpy(output_bg_feature))
        elif bg_generate == 'syn_gt':
            syn_noise_bg.normal_(0, 1)
            output = netG_iou_bg(Variable(syn_noise_bg), Variable(syn_feature_one_batch_bg.cuda()))
            syn_feature_bg.narrow(0, i * num * neg_times, num * neg_times).copy_(output.data.cpu())
        elif bg_generate == 'syn_fg':
            syn_noise_bg.normal_(0, 1)
            syn_feature_one_batch_bg.copy_(output.data.cpu().repeat(neg_times, 1))
            output = netG_iou_bg(Variable(syn_noise_bg), Variable(syn_feature_one_batch_bg.cuda()))
            syn_feature_bg.narrow(0, i * num * neg_times, num * neg_times).copy_(output.data.cpu())

            # syn_noise_bg.normal_(0, 1)
            # syn_feature_one_batch_bg.copy_(output.data.cpu().repeat(neg_times, 1))
            # output = netG_iou_bg(Variable(syn_noise_bg), Variable(syn_feature_one_batch_bg.cuda()))
            # half_part = output.shape[0]/2
            # output = output.data.cpu()[:half_part, :]
            #
            # output_bg_feature = torch.FloatTensor(bg_feature[:neg_times * num/2, :])
            # bg_feature = bg_feature[neg_times * num/2:, :]
            # output_bg_feature = output.new(output.shape).copy_(output_bg_feature)
            # output = torch.cat((output, output_bg_feature), 0)
            #
            # syn_feature_bg.narrow(0, i * num * neg_times, num * neg_times).copy_(output)

        # syn_feature_bg.narrow(0, i * num * neg_times, num * neg_times).copy_(output.data.cpu())
        syn_label_bg.narrow(0, i * num * neg_times, num * neg_times).fill_(0)

    syn_feature = torch.cat((syn_feature, syn_feature_bg), 0)
    syn_label = torch.cat((syn_label, syn_label_bg), 0)

    # iou_inf_bg.copy_(neg_iou[:num*neg_times])
    # neg_iou = neg_iou[num*neg_times:]
    # if neg_iou.shape[0] < num*neg_times:
    #     neg_iou = copy.deepcopy(neg_iou_save)
    #
    # syn_noise_bg.normal_(0, 1)
    # output = netG_iou(Variable(syn_noise_bg), Variable(iou_inf_bg), Variable(syn_feature_one_batch_bg.cuda()))

    return syn_feature, syn_label


def calc_gradient_penalty(netD, real_data, fake_data, input_att, dataset='coco', iou=None):
    # Gradient penalty of WGAN
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    if dataset == 'coco':
        disc_interpolates = netD(interpolates, Variable(input_att))
    else:
        disc_interpolates = netD(interpolates, Variable(input_att), Variable(iou))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty



# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.loss_cls:
    optimizerCls1 = optim.Adam(netCls1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

"added"
optimizerD_iou = optim.Adam(netD_iou.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG_iou = optim.Adam(netG_iou.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

"added"
if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
    optimizerD_iou_bg = optim.Adam(netD_iou_bg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG_iou_bg = optim.Adam(netG_iou_bg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    if opt.loss_cls:
        optimizerCls2 = optim.Adam(netCls2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

time_rem = 0
time_i = 0
# if opt.dataset == 'vg':
#     opt.second_gan = int(opt.total_file_number/opt.file_number) + 1
# elif opt.dataset == 'coco':
    # opt.second_gan = int(7 / opt.file_number) + 1
    # opt.second_gan = 0
    # opt.second_gan = int(3 / opt.file_number) + 1
# if opt.test_code:
#     opt.second_gan = 0
# else:

if opt.test_code == 'no':
    opt.second_gan = int(opt.total_file_number/opt.file_number) + 1
else:
    opt.second_gan = 0
print(opt.test_code)

print('opt.second_gan:' + str(opt.second_gan))

# Start training
for epoch in range(opt.nepoch):
    # set to training mode
    netD.train()
    netG.train()
    netDec.train()

    mean_lossD, mean_lossG = 0, 0
    mean_lossR, mean_lossC = 0, 0
    mean_lossCls = 0

    "added"
    netD_iou.train()
    netG_iou.train()

    if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
        netD_iou_bg.train()
        netG_iou_bg.train()

    mean_lossD_iou, mean_lossG_iou = 0, 0
    mean_lossC_iou = 0
    mean_lossCls_iou = 0

    if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
        mean_lossD_iou_bg, mean_lossG_iou_bg = 0, 0
        mean_lossC_iou_bg = 0
        mean_lossCls_iou_bg = 0

    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective
        ###########################
        begin_time = time.time()
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for p in netD_iou.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
            for p in netD_iou_bg.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()

            # train with realG, sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            # Decoder training
            if opt.loss_rec:
                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = recons_criterion(recons, input_attv)
                R_cost.backward()
                optimizerDec.step()

            if opt.loss_cls:
                netCls1.zero_grad()
                predicted_label = netCls1(input_resv)
                true_label = input_label - 1
                cls_cost = cls_criterion(predicted_label, Variable(true_label))
                cls_cost.backward()
                optimizerCls1.step()

            # Discriminator training with real
            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train Discriminator with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # WGAN gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()
            mean_lossD += D_cost.item()

            "added"
            if epoch >= opt.second_gan:
                netD_iou.zero_grad()
                iouv_inf = Variable(iou_inf)
                # Second discriminator training with real iou_feature
                input_resv_iou = Variable(input_res_iou)
                criticD_real_iou = netD_iou(input_resv_iou, input_attv)
                criticD_real_iou = criticD_real_iou.mean()
                criticD_real_iou.backward(mone)
                #    def forward(self, noise, iou, feature):
                #train second discriminator with fake iou_feature
                noise_iou.normal_(0, 1)
                noisev_iou = Variable(noise_iou)
                # fake_iou = netG_iou(noisev_iou, iouv_inf, fake)
                fake_iou = netG_iou(noisev_iou, fake)
                fake_norm_iou = fake_iou.data[0].norm()
                sparse_fake_iou = fake_iou.data[0].eq(0).sum()
                criticD_fake_iou = netD_iou(fake_iou.detach(), input_attv)
                criticD_fake_iou = criticD_fake_iou.mean()
                criticD_fake_iou.backward(one)

                gradient_penalty_iou = calc_gradient_penalty(netD_iou, input_res_iou, fake_iou.data, input_att, dataset='coco')
                gradient_penalty_iou.backward()

                Wasserstein_D_iou = criticD_real_iou - criticD_fake_iou
                D_cost_iou = criticD_fake_iou - criticD_real_iou + gradient_penalty_iou
                optimizerD_iou.step()
                mean_lossD_iou += D_cost_iou.item()

                if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':

                    netD_iou_bg.zero_grad()
                    iouv_inf_bg = Variable(iou_inf_bg)
                    # Second discriminator training with real iou_feature
                    input_resv_iou_bg = Variable(input_res_iou_bg)

                    if opt.loss_cls:
                        netCls2.zero_grad()
                        predicted_label_iou = netCls2(torch.cat((input_resv_iou,input_resv_iou_bg), 0))
                        label_bg = input_label.new(input_label.shape).zero_()
                        true_label_iou = torch.cat((input_label, label_bg), 0)
                        cls_cost_iou = cls_criterion(predicted_label_iou, Variable(true_label_iou))
                        cls_cost_iou.backward()
                        optimizerCls2.step()


                    criticD_real_iou_bg = netD_iou_bg(input_resv_iou_bg, input_attv)
                    criticD_real_iou_bg = criticD_real_iou_bg.mean()
                    criticD_real_iou_bg.backward(mone)
                    #    def forward(self, noise, iou, feature):
                    # train second discriminator with fake iou_feature
                    noise_iou_bg.normal_(0, 1)
                    noisev_iou_bg = Variable(noise_iou_bg)
                    # fake_iou = netG_iou(noisev_iou, iouv_inf, fake)
                    if opt.bg_generate == 'syn_gt':
                        fake_iou_bg = netG_iou_bg(noisev_iou_bg, fake)
                    elif opt.bg_generate == 'syn_fg':
                        fake_iou_bg = netG_iou_bg(noisev_iou_bg, fake_iou)

                    fake_norm_iou_bg = fake_iou_bg.data[0].norm()
                    sparse_fake_iou_bg = fake_iou_bg.data[0].eq(0).sum()
                    criticD_fake_iou_bg = netD_iou_bg(fake_iou_bg.detach(), input_attv)
                    criticD_fake_iou_bg = criticD_fake_iou_bg.mean()
                    criticD_fake_iou_bg.backward(one)

                    gradient_penalty_iou_bg = calc_gradient_penalty(netD_iou_bg, input_res_iou_bg, fake_iou_bg.data, input_att,
                                                                 dataset='coco')
                    gradient_penalty_iou_bg.backward()

                    Wasserstein_D_iou_bg = criticD_real_iou_bg - criticD_fake_iou_bg
                    D_cost_iou_bg = criticD_fake_iou_bg - criticD_real_iou_bg + gradient_penalty_iou_bg
                    optimizerD_iou_bg.step()
                    mean_lossD_iou_bg += D_cost_iou_bg.item()


        ############################
        # (2) Update G network: optimize WGAN-GP objective
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        for p in netD_iou.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
            for p in netD_iou_bg.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        mean_lossG += G_cost.item()
        errG = G_cost

        ### cosine embedding loss for matching pairs
        if opt.loss_mum:

            temp_label = torch.ones(fake.shape[0])
            if opt.cuda:
                temp_label = temp_label.cuda()
            temp_label = Variable(temp_label)
            # fake and input_resv are matched already
            embed_match = emb_criterion(fake, input_resv, temp_label)

            ### cosine embedding loss for non-matching pairs
            # Randomly permute the labels and real input data
            if opt.cuda:
                rand_index = torch.randperm(input_label.shape[0]).cuda()
            else:
                rand_index = torch.randperm(input_label.shape[0])

            new_label = input_label[rand_index]
            new_feat = input_res[rand_index, :]
            z1 = input_label.cpu().numpy()
            z2 = new_label.cpu().numpy()
            temp_label = -1 * torch.ones(fake.shape[0])
            # Label correction for pairs that remain matched after random permutation
            if len(np.where(z1==z2)[0])>0:
                temp_label[torch.from_numpy(np.where(z1==z2)[0])] = 1
            if opt.cuda:
                temp_label = temp_label.cuda()
            embed_nonmatch = emb_criterion(fake, Variable(new_feat), Variable(temp_label), input_att, input_att[rand_index, :])
            # embed_nonmatch = emb_criterion(fake, Variable(new_feat), Variable(temp_label))
            embed_err = embed_match + embed_nonmatch
            mean_lossC += embed_err.item()
            errG += opt.cosem_weight*embed_err

        ### Attribute reconstruction loss
        if opt.loss_rec:
            netDec.zero_grad()
            recons = netDec(fake)
            R_cost = recons_criterion(recons, input_attv)
            mean_lossR += R_cost.item()
            errG += opt.recons_weight * R_cost

        if opt.loss_cls:
            netCls1.zero_grad()
            predicted_label = netCls1(fake)
            true_label = input_label - 1
            cls_cost = cls_criterion(predicted_label, Variable(true_label))
            mean_lossCls += cls_cost.item()
            errG += opt.cls_weight * cls_cost



        errG.backward(retain_graph=True)
        optimizerG.step()
        # optimizerDec.step()

        "added"
        if epoch >= opt.second_gan:
            netG_iou.zero_grad()
            noise_iou.normal_(0, 1)
            noisev_iou = Variable(noise_iou)
            iouv_inf = Variable(iouv_inf)
            # fake_iou = netG_iou(noisev_iou, iouv_inf, fake)
            fake_iou = netG_iou(noisev_iou, fake)
            criticG_fake_iou = netD_iou(fake_iou, input_attv)
            criticG_fake_iou = criticG_fake_iou.mean()
            G_cost_iou = -criticG_fake_iou
            mean_lossG_iou += G_cost_iou.item()
            errG_iou = G_cost_iou

            if opt.loss_mum:

                temp_label_iou = torch.ones(fake_iou.shape[0])
                if opt.cuda:
                    temp_label_iou = temp_label_iou.cuda()
                temp_label_iou = Variable(temp_label_iou)
                # fake and input_resv are matched already
                embed_match_iou = emb_criterion(fake_iou, input_resv_iou, temp_label_iou)

                ## cosine embedding loss for non-matching pairs
                # Randomly permute the labels and real input data
                if opt.cuda:
                    rand_index_iou = torch.randperm(input_label.shape[0]).cuda()
                else:
                    rand_index_iou = torch.randperm(input_label.shape[0])

                new_label_iou = input_label[rand_index_iou]
                new_feat_iou = input_res_iou[rand_index_iou, :]
                z1_iou = input_label.cpu().numpy()
                z2_iou = new_label_iou.cpu().numpy()
                temp_label_iou = -1 * torch.ones(fake_iou.shape[0])
                # Label correction for pairs that remain matched after random permutation
                if len(np.where(z1_iou == z2_iou)[0]) > 0:
                    temp_label_iou[torch.from_numpy(np.where(z1_iou == z2_iou)[0])] = 1
                if opt.cuda:
                    temp_label_iou = temp_label_iou.cuda()

                embed_nonmatch_iou = emb_criterion(fake_iou, Variable(new_feat_iou), Variable(temp_label_iou), input_att, input_att[rand_index_iou, :])

                # embed_nonmatch_iou = emb_criterion(fake_iou, Variable(new_feat_iou), Variable(temp_label_iou))
                embed_err_iou = embed_match_iou + embed_nonmatch_iou
                mean_lossC_iou += embed_err_iou.item()
                errG_iou += opt.cosem_weight * embed_err_iou

            if opt.loss_cls:
                netCls2.zero_grad()
                predicted_label_iou = netCls2(fake_iou)
                true_label_iou = input_label
                cls_cost_iou = cls_criterion(predicted_label_iou, Variable(true_label_iou))
                mean_lossCls_iou += cls_cost_iou.item()
                errG_iou += opt.cls_weight * cls_cost_iou


            netG.zero_grad()
            errG_iou.backward(retain_graph=True)
            optimizerG_iou.step()
            # optimizerG.step()

            if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':

                netG_iou_bg.zero_grad()
                noise_iou_bg.normal_(0, 1)
                noisev_iou_bg = Variable(noise_iou_bg)
                iouv_inf_bg = Variable(iouv_inf_bg)
                # fake_iou = netG_iou(noisev_iou, iouv_inf, fake)
                if opt.bg_generate == 'syn_gt':
                    fake_iou_bg = netG_iou_bg(noisev_iou_bg, fake)
                elif opt.bg_generate == 'syn_fg':
                    fake_iou_bg = netG_iou_bg(noisev_iou_bg, fake_iou)

                criticG_fake_iou_bg = netD_iou_bg(fake_iou_bg, input_attv)
                criticG_fake_iou_bg = criticG_fake_iou_bg.mean()
                G_cost_iou_bg = -criticG_fake_iou_bg
                mean_lossG_iou_bg += G_cost_iou_bg.item()
                errG_iou_bg = G_cost_iou_bg

                if opt.loss_mum:

                    temp_label_iou_bg = torch.ones(fake_iou.shape[0])
                    if opt.cuda:
                        temp_label_iou_bg = temp_label_iou_bg.cuda()
                    temp_label_iou_bg = Variable(temp_label_iou_bg)
                    # fake and input_resv are matched already
                    embed_match_iou_bg = emb_criterion(fake_iou_bg, input_resv_iou_bg, temp_label_iou_bg)

                    ## cosine embedding loss for non-matching pairs
                    # Randomly permute the labels and real input data
                    if opt.cuda:
                        rand_index_iou_bg = torch.randperm(input_label.shape[0]).cuda()
                    else:
                        rand_index_iou_bg = torch.randperm(input_label.shape[0])

                    new_label_iou_bg = input_label[rand_index_iou_bg]
                    new_feat_iou_bg = input_res_iou[rand_index_iou_bg, :]
                    z1_iou_bg = input_label.cpu().numpy()
                    z2_iou_bg = new_label_iou_bg.cpu().numpy()
                    temp_label_iou_bg = -1 * torch.ones(fake_iou_bg.shape[0])
                    # Label correction for pairs that remain matched after random permutation
                    if len(np.where(z1_iou_bg == z2_iou_bg)[0]) > 0:
                        temp_label_iou_bg[torch.from_numpy(np.where(z1_iou_bg == z2_iou_bg)[0])] = 1
                    if opt.cuda:
                        temp_label_iou_bg = temp_label_iou_bg.cuda()
                    embed_nonmatch_iou_bg = emb_criterion(fake_iou_bg, Variable(new_feat_iou_bg), Variable(temp_label_iou_bg), input_att, input_att[rand_index_iou_bg, :])
                    # embed_nonmatch_iou_bg = emb_criterion(fake_iou_bg, Variable(new_feat_iou_bg), Variable(temp_label_iou_bg))
                    embed_err_iou_bg = embed_match_iou_bg + embed_nonmatch_iou_bg
                    mean_lossC_iou_bg += embed_err_iou_bg.item()
                    errG_iou_bg += opt.cosem_weight * embed_err_iou_bg

                if opt.loss_cls:
                    netCls2.zero_grad()
                    predicted_label_iou_bg = netCls2(fake_iou_bg)
                    label_bg = input_label.new(input_label.shape).zero_()
                    true_label_iou_bg = label_bg
                    cls_cost_iou_bg = cls_criterion(predicted_label_iou_bg, Variable(true_label_iou_bg))
                    mean_lossCls_iou_bg += cls_cost_iou_bg.item()
                    errG_iou_bg += opt.cls_weight1 * cls_cost_iou_bg

                netG.zero_grad()
                errG_iou_bg.backward()
                optimizerG_iou_bg.step()
                # optimizerG.step()

        end_time = time.time()

        time_gap = end_time - begin_time

        time_rem += time_gap
        time_i += 1

        sys.stdout.write('RUN: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i, data.ntrain, time_gap, (time_rem/time_i)*(data.ntrain/opt.batch_size - time_i)))
        sys.stdout.flush()

    time_rem = 0
    time_i = 0

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= opt.critic_iter * data.ntrain / opt.batch_size
    mean_lossC /= data.ntrain / opt.batch_size
    mean_lossR /= data.ntrain / opt.batch_size

    mean_lossG_iou /= data.ntrain / opt.batch_size
    mean_lossD_iou /= opt.critic_iter * data.ntrain / opt.batch_size
    mean_lossC_iou /= data.ntrain / opt.batch_size

    if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':

        mean_lossG_iou_bg /= data.ntrain / opt.batch_size
        mean_lossD_iou_bg /= opt.critic_iter * data.ntrain / opt.batch_size
        mean_lossC_iou_bg /= data.ntrain / opt.batch_size


    if epoch >= opt.second_gan:
        if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f Loss_D_iou: %.4f Loss_G_iou: %.4f, Wasserstein_dist_iou: %.4f Loss_D_iou_bg: %.4f Loss_G_iou_bg: %.4f, Wasserstein_dist_iou_bg: %.4f' % (epoch, opt.nepoch, mean_lossD, mean_lossG, Wasserstein_D.item(),
                                                                                                                                                                                                                  mean_lossD_iou, mean_lossG_iou, Wasserstein_D_iou.item(), mean_lossD_iou_bg, mean_lossG_iou_bg, Wasserstein_D_iou_bg.item()))
        else:
            print(
                '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f Loss_D_iou: %.4f Loss_G_iou: %.4f, Wasserstein_dist_iou: %.4f' % (
                epoch, opt.nepoch, mean_lossD, mean_lossG, Wasserstein_D.item(), mean_lossD_iou, mean_lossG_iou, Wasserstein_D_iou.item()))

    else:
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f ' % (epoch, opt.nepoch, mean_lossD, mean_lossG, Wasserstein_D.item()))



    # set to evaluation mode
    netG.eval()
    netDec.eval()

    # Synthesize unseen class samples
    if opt.dataset == 'coco':
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, torch.from_numpy(data.attributes), opt.syn_num)
    elif opt.dataset == 'vg' or opt.dataset == 'imagenet':
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses[1:], torch.from_numpy(data.unseen_attributes), opt.syn_num)
    if opt.gzsl_od:
        # OD based GZSL
        seen_class = data.seenclasses.size(0)
        clsu = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, _nepoch=25, _batch_size=opt.syn_num)
        clss = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label,data.seenclasses), data, seen_class, opt.cuda, _nepoch=25, _batch_size=opt.syn_num, test_on_seen=True)
        clsg = classifier_entropy.CLASSIFIER(data.train_feature, util.map_label(data.train_label,data.seenclasses), data, seen_class, syn_feature, syn_label, opt.cuda, clss, clsu, _batch_size=128)
        print('GZSL-OD: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))
    elif opt.gzsl:
        # Generalized zero-shot learning
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        clsg = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, _nepoch=25, _batch_size=opt.syn_num, generalized=True)
        print('GZSL: Acc seen=%.4f, Acc unseen=%.4f, h=%.4f' % (clsg.acc_seen, clsg.acc_unseen, clsg.H))
    else:
        # Zero-shot learning
        if opt.dataset == 'coco':
            clsz = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, _nepoch=25, _batch_size=opt.syn_num, dataset=opt.dataset)
        elif opt.dataset == 'vg' or opt.dataset == 'imagenet':
            clsz = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses[1:]), data, data.unseenclasses[1:].size(0), opt.cuda, _nepoch=25, _batch_size=opt.syn_num, dataset=opt.dataset)
        print("")
        print('ZSL: Acc unseen=%.4f' % (clsz.acc))

    "added"
    if epoch >= opt.second_gan:
        netG_iou.eval()

        # if opt.bg_generate == 'syn_gt' or opt.bg_generate == 'syn_fg':
        #     netG_iou_bg.eval()
        #
        # iou_sample_number = data.unseenclasses.size(0) * opt.syn_num
        # if opt.dataset == 'coco':
        #     file_path = "/root/Workspace/zsz/gan_training_file/training_feature_48seen_cag_wogt/fg_feature/fg_feature1/fg_feature1.pkl"
        # elif opt.dataset == 'vg':
        #     file_path = "/root/Workspace/zsz/gan_training_file_vg/train_feature/fg_feature/fg_feature1/fg_feature13.pkl"
        # with open(file_path, 'rb') as f:
        #     feature_set = pickle.load(f)
        # feature_set_iou = feature_set[:, 1].reshape(-1, 1)
        # index = np.random.permutation(feature_set_iou.shape[0])
        # pos_iou = torch.from_numpy(feature_set_iou[index[:iou_sample_number]]).repeat(1, opt.iou_information_size)
        #
        # neg_iou = []
        # for neg_indx in range(10):
        #
        #     if opt.dataset == 'coco':
        #         head = '/root/Workspace/zsz/gan_training_file/training_feature_48seen_cag_wogt/bg_feature/bg_feature'
        #     elif opt.dataset == 'vg':
        #         head = '/root/Workspace/zsz/gan_training_file_vg/train_feature/bg_feature/bg_feature'
        #     head +=  str(neg_indx) + '/bg_feature' + str(neg_indx) + '.pkl'
        #
        #     with open(file_path, 'rb') as f:
        #         feature_set = pickle.load(f)
        #     feature_set_iou = feature_set[:, 1]
        #     index = np.random.permutation(feature_set_iou.shape[0])
        #     one_neg_iou = torch.from_numpy(feature_set_iou[index[:iou_sample_number * 2]]).reshape(-1, 1)
        #     neg_iou.append(one_neg_iou)
        # neg_iou = torch.cat(neg_iou, 0).repeat(1, opt.iou_information_size)

        pos_iou = []
        neg_iou = []

        if opt.bg_generate == 'seen':
            netG_iou_bg = None

        if opt.dataset == 'coco':

            syn_feature, syn_label = generate_syn_feature_iou(netG, data.unseenclasses, torch.from_numpy(data.attributes),
                                                              opt.syn_num, netG_iou, pos_iou=pos_iou, neg_iou=neg_iou,
                                                              neg_times=opt.neg_times, bg_generate=opt.bg_generate, netG_iou_bg=netG_iou_bg)

            #(netG, classes, attribute, num, netG_iou, pos_iou, neg_iou, neg_times=3)
            clsz = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses_bg), data,
                                         data.unseenclasses_bg.size(0), opt.cuda, _nepoch=25, _batch_size=opt.syn_num, bg=True, dataset=opt.dataset)
        elif opt.dataset == 'vg' or opt.dataset == 'imagenet':
            syn_feature, syn_label = generate_syn_feature_iou(netG, data.unseenclasses[1:],
                                                              torch.from_numpy(data.unseen_attributes),
                                                              opt.syn_num, netG_iou, pos_iou=pos_iou, neg_iou=neg_iou,
                                                              neg_times=opt.neg_times, bg_generate=opt.bg_generate, netG_iou_bg=netG_iou_bg, dataset=opt.dataset)

            # (netG, classes, attribute, num, netG_iou, pos_iou, neg_iou, neg_times=3)
            clsz = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                         data.unseenclasses.size(0), opt.cuda, _nepoch=1, _batch_size=opt.syn_num,
                                         bg=True, dataset=opt.dataset)
        print("")
        print('ZSL: Acc unseen=%.4f' % (clsz.acc))


    output_dir = '/root/Workspace/zsz/faster-rcnn-3/gzsl-od-master/MODEL/' + opt.sv
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, 'clswgan_{}.pth'.format(epoch))
    state = {
        'epoch': epoch + 1,
        'des': netD.state_dict(),
        'gen': netG.state_dict(),
        'dec': netDec.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerDec': optimizerDec.state_dict(),
        'cls_layer': clsz.best_model.state_dict()
    }
    torch.save(state, filename)

    data.reset_training_data(opt)


     
