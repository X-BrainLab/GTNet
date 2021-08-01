import torch.nn as nn
import torch
from torch.autograd import Variable
import copy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Res_Linear') == -1:
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_CRITIC_iou(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC_iou, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize + opt.iou_information_size, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att, iou):
        h = torch.cat((x, att, iou), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class Res_Linear(nn.Module):
    def __init__(self, opt):
        super(Res_Linear, self).__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        # self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, ngh):
        h1 = ngh
        h = self.relu(self.fc1(ngh))
        h = self.relu(self.fc2(h))
        return h + h1

class MLP_G(nn.Module):
    def __init__(self, opt, use_relu=False):
        super(MLP_G, self).__init__()
        self.use_relu = use_relu
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        # if self.use_relu:
        #     self.res_fc = Res_Linear(opt)
        # self.fca1 = nn.Linear(opt.ngh, opt.ngh)
        # self.fca2 = nn.Linear(opt.ngh, opt.ngh)

        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        # if self.use_relu:
        #     h = self.res_fc(h)
        # h = self.relu(self.fca1(h))
        # h = self.relu(self.fca2(h))
        h = self.relu(self.fc3(h))
        return h

class MLP_G_2nd(nn.Module):
    def __init__(self, opt):
        super(MLP_G_2nd, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.nz + opt.attSize , opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, res, iou):
        h = torch.cat((noise, res, iou), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class Dec(nn.Module):
    def __init__(self, opt):
        super(Dec, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, feat):      
        h = self.lrelu(self.fc1(feat))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h

class MLP_G_IOU(nn.Module):
    def __init__(self, opt):
        super(MLP_G_IOU, self).__init__()
        self.fc1 = nn.Linear(opt.nz + opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, feature):
        h = torch.cat((noise, feature), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class cls_loss_layer(nn.Module):
    def __init__(self, opt, class_number):
        super(cls_loss_layer, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, class_number)
        self.apply(weights_init)

    def forward(self, feature):
        h = self.fc1(feature)
        return h



# class Dec_IOU(nn.Module):
#     def __init__(self, opt):
#         super(Dec_IOU, self).__init__()
#         self.fc1 = nn.Linear(opt.resSize, opt.ngh)
#         self.fc2 = nn.Linear(opt.ngh, opt.ngh)
#         self.fc3 = nn.Linear(opt.ngh, opt.resSize)
#         self.fc4 = nn.Linear(opt.resSize, opt.ngh)
#         self.fc5 = nn.Linear(opt.ngh, opt.ngh)
#         self.fc6 = nn.Linear(opt.ngh, 1)
#         self.lrelu = nn.LeakyReLU(0.2, True)
#         self.relu = nn.ReLU(True)
#         self.apply(weights_init)
#
#     def forward(self, feat):
#         h = self.lrelu(self.fc1(feat))
#         h = self.lrelu(self.fc2(h))
#         h = self.fc3(h)
#         latent_embedding = copy.deepcopy(h.detach().cpu().numpy())
#         h = self.lrelu(self.fc4(h))
#         h = self.lrelu(self.fc5(h))
#         h = self.fc6(h)
#         return h, latent_embedding

class Dec_IOU(nn.Module):
    def __init__(self, opt):
        super(Dec_IOU, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, feat):
        h = self.lrelu(self.fc1(feat))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h


class MLP_G_baseline(nn.Module):
    def __init__(self, opt):
        super(MLP_G_baseline, self).__init__()
        self.fc1 = nn.Linear(opt.nz + opt.iou_information_size + opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, iou, att):
        h = torch.cat((noise, iou, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class MLP_G_baseline_Dec(nn.Module):
    def __init__(self, opt):
        super(MLP_G_baseline_Dec, self).__init__()
        self.fc1 = nn.Linear(opt.nz + opt.iou_information_size + opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, iou, att):
        h = torch.cat((noise, iou, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h







    
