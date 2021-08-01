import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
import copy
import pickle
import torch.nn.functional as F
import sys


class CLASSIFIER:
    # train_Y is interger
    def __init__(self, _cuda=True, _batch_size=100,test_on_seen=False):

        self.test_seen_feature, self.test_seen_label = self.load_test_seen_feature()
        self.seenclasses = self.load_seen_classes()

        """"""
        self.batch_size = _batch_size
        # self.nepoch = _nepoch
        self.nclass = self.seenclasses.shape[0]
        self.input_dim = self.test_seen_feature.shape[1]
        self.cuda = _cuda
        self.model = CLS_LAYER(self.input_dim, self.nclass)

        # self.criterion = nn.NLLLoss()
        # self.criterion = F.cross_entropy()

        # self.model.apply(util.weights_init)

        # self.input = torch.FloatTensor(_batch_size, self.input_dim)
        # self.label = torch.LongTensor(_batch_size)

        # self.lr = _lr
        # self.beta1 = _beta1
        # setup optimizer
        # self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.test_on_seen = test_on_seen

        if self.cuda:
            self.model.cuda()

        self.acc, self.best_model = self.pretrained_seen_model()

    def load_test_seen_feature(self):
        # head = '/root/Workspace/zsz/gan_training_file/val/test_seen_features_cag/'
        # test_seen_feature = []
        # for indx in range(3):
        #     file_name = 'feature' + str(indx*100) + '.pkl'
        #     path = head + file_name
        #     with open(path, 'rb') as f:
        #         feature_set = pickle.load(f)
        #     test_seen_feature.append(feature_set)
        #
        # test_seen_feature = np.concatenate(test_seen_feature, axis=0)
        # test_seen_label = test_seen_feature[:, 2].astype(np.int16)
        # test_seen_feature = test_seen_feature[:, 3:]
        head = '/root/Workspace/zsz/gan_training_file/training_feature_48seen_cag_wogt/'

        folder_number = 4
        file_number = 4
        fg_feature_set = [[] for i in range(folder_number)]
        file_loading = 0
        for ind1 in range(folder_number):
            for ind2 in range(file_number):
                path = head + 'fg_feature/' + 'fg_feature' + str(ind1) + '/fg_feature' + str(ind2) + '.pkl'
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                fg_feature_set[ind1].append(feature_set)
                file_loading += 1
                sys.stdout.write('Loading FG dataset {:d}:  {:d}/{:d} \r'.format(feature_set.shape[0], file_loading,
                                                                                 folder_number * file_number))
                sys.stdout.flush()
            fg_feature_set[ind1] = np.concatenate(fg_feature_set[ind1], axis=0)
        fg_feature_set = np.concatenate(fg_feature_set, axis=0)

        file_loading = 0
        bg_feature_set = [[] for i in range(folder_number)]
        for ind1 in range(folder_number):
            for ind2 in range(file_number):
                path = head + 'bg_feature/' + 'bg_feature' + str(ind1) + '/bg_feature' + str(ind2) + '.pkl'
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                bg_feature_set[ind1].append(feature_set)
                file_loading += 1
                sys.stdout.write('Loading dataset BG {:d}:  {:d}/{:d} \r'.format(feature_set.shape[0], file_loading,
                                                                                 folder_number * file_number))
                sys.stdout.flush()
            bg_feature_set[ind1] = np.concatenate(bg_feature_set[ind1], axis=0)
        bg_feature_set = np.concatenate(bg_feature_set, axis=0)
        bg_feature_set[:, 2] = 0

        batch_fg_feature = fg_feature_set[:, 3:]
        batch_bg_feature = bg_feature_set[:, 3:]

        batch_fg_label = fg_feature_set[:, 2].astype(np.int16).reshape(-1, )
        batch_bg_label = bg_feature_set[:, 2].astype(np.int16).reshape(-1, )

        test_seen_feature = np.concatenate((batch_fg_feature, batch_bg_feature), axis=0)
        test_seen_label = np.concatenate((batch_fg_label, batch_bg_label), axis=0)

        return torch.from_numpy(test_seen_feature), torch.from_numpy(test_seen_label)

    def load_seen_classes(self):
        head = '/root/Workspace/zsz/gan_training_file/train_seen/'
        seen_class_file = 'id_to_cocoid.pkl'
        file_path = head + seen_class_file
        seen_classes = torch.from_numpy(self.load_pickle_file(file_path).astype(np.int16)).reshape(-1, 1)

        return seen_classes

    def load_pickle_file(self, path):
        with open(path, 'rb') as f:
            content = pickle.load(f)
        return content

    def pretrained_seen_model(self):
        print("loading the classification layer from faster-rcnn")
        load_name = '/root/Workspace/zsz/faster-rcnn-3/models/coco_48seen_baseline/res101/coco/faster_rcnn_1_10_7714.pth'
        checkpoint = torch.load(load_name)
        in_dict = ['RCNN_cls_score.weight', 'RCNN_cls_score.bias']
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in in_dict}
        self.model.load_state_dict(pretrained_dict)
        print("loading finished")
        best_model = copy.deepcopy(self.model)
        acc = self.val_zsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)

        return best_model, acc

    def val_zsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        test_data = torch.FloatTensor(0).cuda()
        test_data = Variable(test_data)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            test_data.data.resize_(test_X[start:end, :].size()).copy_(test_X[start:end, :])
            if self.cuda:
                output = self.model(test_data)
            else:
                output = self.model(Variable(test_X[start:end], volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_zsl(util.map_label(test_label, target_classes), predicted_label,
                                             target_classes.size(0))
        return acc

    def compute_per_class_acc_zsl(self, test_label, predicted_label, nclass):
        # test_label = test_label.numpy()
        # predicted_label = predicted_label.numpy()
        # acc_per_class = torch.FloatTensor(nclass).fill_(0)
        # for i in range(nclass):
        #     idx = (test_label == i)
        #     if np.sum(idx) < 1:
        #         continue
        #     acc_per_class[i] = np.sum(test_label[idx] == predicted_label[idx]).astype(np.float) / np.sum(idx).astype(np.float)
        # acc_per_class = acc_per_class[1:]
        # print(acc_per_classt)
        # return acc_per_class.mean()

        test_label = test_label.numpy()
        predicted_label = predicted_label.numpy()
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        class_num = 0
        for i in range(nclass):
            idx = (test_label == i)
            if np.sum(idx) < 1:
                continue
            class_num += 1
            acc_per_class[i] = np.sum(test_label[idx] == predicted_label[idx]).astype(np.float) / np.sum(idx).astype(
                np.float)
        print(acc_per_class.sum() / class_num)
        return acc_per_class.sum() / class_num



class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class CLS_LAYER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(CLS_LAYER, self).__init__()
        self.RCNN_cls_score = nn.Linear(input_dim, nclass)

    def forward(self, x):
        o = self.RCNN_cls_score(x)
        return o


if __name__ == '__main__':

    # OD based GZSL
    clss = CLASSIFIER(_batch_size=100)




