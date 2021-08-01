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
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=False, test_on_seen=False, bg=False, dataset='coco'):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.dataset = dataset
        print(self.dataset)
        if test_on_seen:
            self.test_seen_feature = data_loader.test_seen_feature
            self.test_seen_label = data_loader.test_seen_label
            self.seenclasses = data_loader.seenclasses
        else:
            if bg:
                self.test_unseen_feature, self.test_unseen_label = self.load_test_unseen_feature(bg=bg)
                if self.dataset == 'coco':
                    self.unseenclasses = data_loader.unseenclasses_bg
                elif self.dataset == 'vg' or self.dataset == 'imagenet':
                    self.unseenclasses = data_loader.unseenclasses

            else:
                self.test_unseen_feature, self.test_unseen_label = self.load_test_unseen_feature()
                if self.dataset == 'coco':
                    self.unseenclasses = data_loader.unseenclasses
                elif self.dataset == 'vg' or self.dataset == 'imagenet':
                    self.unseenclasses = data_loader.unseenclasses[1:]

        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        # self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        if test_on_seen:
            self.model = CLS_LAYER(self.input_dim, self.nclass)
        else:
            self.model = CLS_LAYER(self.input_dim, self.nclass)

        self.criterion = nn.NLLLoss()
        # self.criterion = F.cross_entropy()
        self.bg = bg
        self.model.apply(util.weights_init)

        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.test_on_seen = test_on_seen

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.best_model = self.fit_gzsl()
        else:
            self.acc, self.best_model = self.fit_zsl()

    def load_test_unseen_feature(self, bg=False):

        path = '/root/Workspace/zsz/gan_training_file/val/id_to_cocoid.pkl'

        with open(path, 'rb') as f:
            id_to_cocoid = pickle.load(f)


        test_file_number = 1


        if not bg:
            if self.dataset == 'coco':
                head = '/root/Workspace/zsz/gan_training_file_coco/test_feature/gt_feature/'
            elif self.dataset == 'vg':
                head = '/root/Workspace/zsz/gan_training_file_vg/test_feature/gt_feature/'
            elif self.dataset == 'imagenet':
                head = '/root/Workspace/zsz/gan_training_file_imagenet/testing_feature/gt_feature/'

            test_unseen_feature = []
            for indx in range(test_file_number):
                file_name = 'gt_feature' + str(indx) + '.pkl'
                path = head + file_name
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                test_unseen_feature.append(feature_set)
            test_unseen_feature = np.concatenate(test_unseen_feature, axis=0)
            if self.dataset == 'coco':
                test_unseen_label = id_to_cocoid[test_unseen_feature[:, 2].astype(np.int16)].astype(np.int16).reshape(-1, )
            elif self.dataset == 'vg' or self.dataset == 'imagenet':
                test_unseen_label = test_unseen_feature[:, 2].astype(np.int16).reshape(-1, )

            test_unseen_feature = test_unseen_feature[:, 3:]
        else:
            if self.dataset == 'coco':
                head = '/root/Workspace/zsz/gan_training_file_coco/test_feature/fg_feature/fg_feature0/'
                # head = '/root/Workspace/zsz/gan_training_file_coco_test/fg_feature/fg_feature0/'
            elif self.dataset == 'vg':
                head = '/root/Workspace/zsz/gan_training_file_vg/test_feature/fg_feature/fg_feature1/'
            elif self.dataset == 'imagenet':
                head = '/root/Workspace/zsz/gan_training_file_imagenet/testing_feature/fg_feature/fg_feature1/'

            test_unseen_feature_fg = []
            for indx in range(test_file_number):
                file_name = 'fg_feature' + str(indx) + '.pkl'
                path = head + file_name
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                test_unseen_feature_fg.append(feature_set)
            test_unseen_feature_fg = np.concatenate(test_unseen_feature_fg, axis=0)
            if self.dataset == 'coco':
                test_unseen_label_fg = id_to_cocoid[test_unseen_feature_fg[:, 2].astype(np.int16)].astype(np.int16).reshape(-1, )
            elif self.dataset == 'vg' or self.dataset == 'imagenet':
                test_unseen_label_fg = test_unseen_feature_fg[:, 2].astype(np.int16).reshape(-1, )
            test_unseen_feature_fg = test_unseen_feature_fg[:, 3:]

            self.test_unseen_feature_fg_num = test_unseen_feature_fg.shape[0]



            if self.dataset == 'coco':
                head = '/root/Workspace/zsz/gan_training_file_coco/test_feature/bg_feature/bg_feature0/'
                # head = '/root/Workspace/zsz/gan_training_file_coco_test/bbg_feature/'
            elif self.dataset == 'vg':
                head = '/root/Workspace/zsz/gan_training_file_vg/test_feature/bg_feature/bg_feature1/'
            elif self.dataset == 'imagenet':
                head = '/root/Workspace/zsz/gan_training_file_imagenet/testing_feature/bg_feature/bg_feature1/'

            test_unseen_feature_bg = []
            for indx in range(test_file_number):
                file_name = 'bg_feature' + str(indx) + '.pkl'
                path = head + file_name
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                test_unseen_feature_bg.append(feature_set)
            test_unseen_feature_bg = np.concatenate(test_unseen_feature_bg, axis=0)
            test_unseen_label_bg = test_unseen_feature_bg[:, 2].astype(np.int16)

            # if self.dataset == 'coco':
            #     test_unseen_label_bg = id_to_cocoid[test_unseen_feature_bg[:, 2].astype(np.int16)].astype(np.int16).reshape(-1, )
            # elif self.dataset == 'vg':
            #     test_unseen_label_bg = test_unseen_feature_bg[:, 2].astype(np.int16).reshape(-1, )

            test_unseen_label_bg = np.zeros(test_unseen_label_bg.shape[0]).astype(np.int16)
            test_unseen_feature_bg = test_unseen_feature_bg[:, 3:]

            test_unseen_label = np.concatenate((test_unseen_label_fg, test_unseen_label_bg), axis=0)
            test_unseen_feature = np.concatenate((test_unseen_feature_fg, test_unseen_feature_bg), axis=0)

        return torch.from_numpy(test_unseen_feature), torch.from_numpy(test_unseen_label)

    def fit_zsl(self):
        best_acc = 0
        acc_fg = 0
        acc_bg = 0
        acc1 = 0
        acc2 = 0
        max_epoch = 0
        best_model = copy.deepcopy(self.model)
        for epoch in range(self.nepoch):
            self.model.train()
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = F.cross_entropy(output, labelv)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            if self.test_on_seen:
                acc = self.val_zsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            else:
                acc = self.val_zsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
                if self.bg:
                    feature_num = self.test_unseen_feature_fg_num
                    acc1 = self.val_zsl(self.test_unseen_feature[:feature_num, :], self.test_unseen_label[:feature_num], self.unseenclasses)
                    acc2 = self.val_zsl(self.test_unseen_feature[feature_num:, :], self .test_unseen_label[feature_num:], self.unseenclasses)
            if self.dataset == 'coco':
                if acc1 > best_acc:
                    best_acc = acc1
                    acc_fg = acc1
                    acc_bg = acc2
                    # best_model = copy.deepcopy(self.model)
                    max_epoch = epoch
            else:
                if acc > best_acc:
                    best_acc = acc
                    acc_fg = acc1
                    acc_bg = acc2
                    # best_model = copy.deepcopy(self.model)
                    max_epoch = epoch

            best_model = copy.deepcopy(self.model)

            # sys.stdout.write('training zero-shot learing model: {:d}/{:d}   acc:{:.4f}  acc_fg:{:.4f}  acc_bg:{:.4f}\r'.format(epoch, self.nepoch, best_acc, acc_fg, acc_bg))
            # sys.stdout.flush()
            print('training zero-shot learing model: {:d}/{:d}   acc:{:.4f}  acc_fg:{:.4f}  acc_bg:{:.4f}\r'.format(epoch, self.nepoch,  acc, acc1, acc2))
        print('max_epoch: ' + str(max_epoch))
        return best_acc, best_model

    def fit_gzsl(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            self.model.train()
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv) 
                loss.backward()
                self.optimizer.step()
            acc_seen, acc_unseen = 0, 0
            self.model.eval()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            eps = 1e-12
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen+eps)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                best_model = copy.deepcopy(self.model)
                
        return best_seen, best_unseen, best_H, best_model
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm, :]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data before beginning the next epoch
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]            
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_val = torch.FloatTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True)) 
            else:
                output = self.model(Variable(test_X[start:end], volatile=True)) 
            predicted_val[start:end], predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class

    # test_label is integer
    def val_zsl(self, test_X, test_label, target_classes): 
        # start = 0
        # ntest = test_X.size()[0]
        # predicted_label = torch.LongTensor(test_label.size())
        # for i in range(0, ntest, self.batch_size):
        #     end = min(ntest, start+self.batch_size)
        #     if self.cuda:
        #         output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
        #     else:
        #         output = self.model(Variable(test_X[start:end], volatile=True))
        #     _, predicted_label[start:end] = torch.max(output.data, 1)
        #     start = end

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

        acc = self.compute_per_class_acc_zsl(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))

        return acc

    def compute_per_class_acc_zsl(self, test_label, predicted_label, nclass):
        # acc_per_class = torch.FloatTensor(nclass).fill_(0)
        # for i in range(nclass):
        #     idx = (test_label == i)
        #     acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
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
        return acc_per_class.sum()/class_num

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
        
