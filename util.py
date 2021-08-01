import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import pickle
from torch.autograd import Variable
from multiprocessing import Pool
import time
import sys

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_feature(ind):
    head = '/root/Workspace/zsz/gan_training_file/train_seen/'
    file_path = head + 'single_line_feature/feature' + str(ind) + '.pkl'
    with open(file_path, 'rb') as f:
        feature_set = pickle.load(f)
    return feature_set.reshape(1, -1)

def save_model(save_name):
    output_dir = '/root/Workspace/zsz/faster-rcnn-3/gzsl-od-master/model/' + save_name + '/'


def map_label(label, classes):

    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()
        
class DATA_LOADER(object):
    def __init__(self, opt):
        assert opt.matdataset, 'Can load dataset in MATLAB format only'
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        # load features and labels
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.action_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        # Load split details
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/split_" + str(opt.split) + "/" + opt.class_embedding + "_splits.mat")

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()

                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    # Random batch sampling
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att


    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[:batch_size], iclass_label[:batch_size], self.attribute[iclass_label[:batch_size]]


class Data_Loader(object):

    def __init__(self, opt):
        # self.read_matdataset(opt)
        # if opt.dataset == 'coco':
        #     self.total_file_number = 3
        # elif opt.dataset == 'vg':
        #     self.total_file_number = 23
        self.total_file_number = opt.total_file_number

        self.total_file_number_list = range(opt.total_file_number)*50
        print(range(self.total_file_number))
        self.file_start = 0

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.no_bg_file = opt.no_bg_file
        self.file_number = 1
        self.head = '/root/Workspace/zsz/gan_training_file/train_seen/'
        self.number_file = 'seen_feature_number.pkl'
        file_path = self.head + self.number_file
        feature_number = self.load_pickle_file(file_path)
        self.ntrain = 0
        self.gt_num = 0
        self.all_feature_number = feature_number['all_feature']


        # self.fg_bg_feature_set = np.concatenate((self.fg_feature_set, self.bg_feature_set), axis=0)
        # self.ntrain *= self.fg_bg_feature_set.shape[0]
        self.dataset = opt.dataset

        if opt.dataset == 'coco':
            self.attribute_file = 'cocoid_to_attr.pkl'

            # print(self.attribute_file)
            file_path = self.head + self.attribute_file
            self.attributes = self.load_pickle_file(file_path)

            self.seen_class_file = 'id_to_cocoid.pkl'
            file_path = self.head + self.seen_class_file
            self.seenclasses = self.load_pickle_file(file_path).astype(np.int16)

            file_path = '/root/Workspace/zsz/gan_training_file/val/id_to_cocoid.pkl'
            self.unseenclasses = self.load_pickle_file(file_path).astype(np.int16)

            self.unseenclasses = torch.from_numpy(np.setdiff1d(self.unseenclasses, self.seenclasses)).reshape(-1, 1)

            bg = torch.ShortTensor([[0]])
            self.unseenclasses_bg = torch.cat((bg, self.unseenclasses), 0)

            self.seenclasses = torch.from_numpy(self.seenclasses).reshape(-1, 1)

        elif opt.dataset == 'vg':
            file_path = '/root/Workspace/zsz/data/vg_data/vg/seen_unseen_class/seen_cls_to_wordvector.pkl'
            self.attributes = self.load_pickle_file(file_path)

            self.seenclasses = torch.from_numpy(np.array(range(self.attributes.shape[0])).astype(np.int16))

            file_path = '/root/Workspace/zsz/data/vg_data/vg/seen_unseen_class/unseen_cls_to_wordvector.pkl'
            self.unseen_attributes = self.load_pickle_file(file_path)


            self.unseenclasses = torch.from_numpy(np.array(range(self.unseen_attributes.shape[0])).astype(np.int16))
        elif opt.dataset == 'imagenet':
            file_path = '/root/Workspace/zsz/gan_training_file_imagenet/class_information/seen_class_ind_to_attri1.pkl'
            self.attributes = self.load_pickle_file(file_path)

            self.seenclasses = torch.from_numpy(np.array(range(self.attributes.shape[0])).astype(np.int16))

            file_path = '/root/Workspace/zsz/gan_training_file_imagenet/class_information/unseen_class_ind_to_attri1.pkl'
            self.unseen_attributes = self.load_pickle_file(file_path)

            self.unseenclasses = torch.from_numpy(np.array(range(self.unseen_attributes.shape[0])).astype(np.int16))


        if self.dataset == 'vg':
            self.reset_training_data(opt)
        elif self.dataset == 'imagenet':
            self.reset_training_data(opt)
        elif self.dataset == 'coco':
            # self.file_number_list = range(opt.file_number)
            # self.gt_feature_set, self.fg_feature_set, self.bg_feature_set = self.load_feature_file(opt, self.file_number_list)
            self.reset_training_data(opt)


        self.ntrain_class = self.seenclasses.shape[0]
        self.ntest_class = self.unseenclasses.shape[0]
        self.index = np.random.permutation(range(self.ntrain))

        # self.index_for_gt = np.random.permutation(range(self.gt_num))

    def load_pickle_file(self, path):
        with open(path, 'rb') as f:
            content = pickle.load(f)
        return content

    def reset_training_data(self, opt):
        if opt.file_number < self.total_file_number:
            self.ntrain = 0

            self.file_number_list = self.total_file_number_list[self.file_start: self.file_start + opt.file_number]

            self.gt_feature_set, self.fg_feature_set, self.bg_feature_set = self.load_feature_file(opt, self.file_number_list)
            self.file_start += opt.file_number
            print('loading data again' + str(self.file_number_list))


    def load_feature_file(self, opt, file_number_list, load_for_onegan=True):
        beg = time.time()

        if opt.dataset=='coco':
            # head = '/root/Workspace/zsz/gan_training_file/training_feature_48seen_cag_wogt/'
            # head = '/root/Workspace/zsz/gan_training_file_coco/'
            # head = '/root/Workspace/zsz/gan_training_file_coco/train_feature/'
            head = '/root/Workspace/zsz/gan_training_file_coco/train_feature/'
        elif opt.dataset=='vg':
            head = '/root/Workspace/zsz/gan_training_file_vg/train_feature/'
        elif opt.dataset == 'imagenet':
            head = '/root/Workspace/zsz/gan_training_file_imagenet/training_feature/'
        # all_gt_feature_set = []
        # for ind in range(opt.all_gt_file):
        #     path = head + 'all_gt_feature/all_gt_feature' + str(ind) + '.pkl'
        #     with open(path, 'rb') as f:
        #         feature_set = pickle.load(f)
        #     all_gt_feature_set.append(feature_set)
        #     self.gt_num += feature_set.shape[0]
        #     sys.stdout.write('Loading GT dataset {:d}:  {:d}/{:d} \r'.format(feature_set.shape[0], ind+1, opt.all_gt_file))
        #     sys.stdout.flush()
        # all_gt_feature_set = np.concatenate(all_gt_feature_set, axis=0)
        self.gt_num = 0

        gt_feature_set = []
        for ind in file_number_list:
            path = head + 'gt_feature/gt_feature' + str(ind) + '.pkl'
            with open(path, 'rb') as f:
                feature_set = pickle.load(f)
            gt_feature_set.append(feature_set)
            self.gt_num += feature_set.shape[0]
            sys.stdout.write(
                'Loading GT dataset {:d}:  {:d}/{:d} \r'.format(feature_set.shape[0], ind + 1, opt.file_number))
            sys.stdout.flush()
        gt_feature_set = np.concatenate(gt_feature_set, axis=0)

        fg_feature_set = [[] for i in range(opt.folder_number)]
        file_loading = 0
        for ind1 in range(opt.folder_number):
            for ind2 in file_number_list:
                path = head + 'fg_feature/' + 'fg_feature' + str(ind1) + '/fg_feature' + str(ind2) + '.pkl'
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                fg_feature_set[ind1].append(feature_set)
                self.ntrain += feature_set.shape[0]
                file_loading += 1
                sys.stdout.write('Loading FG dataset {:d}:  {:d}/{:d} \r'.format(feature_set.shape[0], file_loading, opt.folder_number*opt.file_number))
                sys.stdout.flush()
            fg_feature_set[ind1] = np.concatenate(fg_feature_set[ind1], axis=0)
        fg_feature_set = np.concatenate(fg_feature_set, axis=0)

        # gt_num = gt_feature_set.shape[0]
        # fg_feature_set[:gt_num, :] = gt_feature_set
        # print("with gt")

        file_loading = 0
        bg_feature_set = [[] for i in range(opt.folder_number)]
        for ind1 in range(opt.folder_number):
            for ind2 in file_number_list:
                path = head + 'bg_feature/' + 'bg_feature' + str(ind1) + '/bg_feature' + str(ind2) + '.pkl'
                with open(path, 'rb') as f:
                    feature_set = pickle.load(f)
                bg_feature_set[ind1].append(feature_set)
                file_loading += 1
                sys.stdout.write('Loading dataset BG {:d}:  {:d}/{:d} \r'.format(feature_set.shape[0], file_loading, opt.folder_number*opt.file_number))
                sys.stdout.flush()
            bg_feature_set[ind1] = np.concatenate(bg_feature_set[ind1], axis=0)
        bg_feature_set = np.concatenate(bg_feature_set, axis=0)

        end = time.time()

        print(str(end-beg)+'s '+'for loading dataset')
        print('GT: ' + str(gt_feature_set.shape))
        print('FG: ' + str(fg_feature_set.shape))
        print('BG: ' + str(bg_feature_set.shape))

        return gt_feature_set, fg_feature_set, bg_feature_set

    def next_batch(self, batch_size):

        if self.index.shape[0] < batch_size:
            self.index = np.random.permutation(range(self.ntrain*2))

        # if self.index_for_gt.shape[0] < batch_size:
        #     self.index_for_gt = np.random.permutation(range(self.gt_num))
        # indx_for_gt = self.index_for_gt[:batch_size].tolist()
        # self.index_for_gt = self.index_for_gt[batch_size:]

        indx = self.index[:batch_size].tolist()
        self.index = self.index[batch_size:]

        indx = np.mod(indx, self.ntrain)
        indx1 = indx[:batch_size/2]
        indx2 = indx[batch_size/2:]

        # batch_fg_feature_info = self.fg_feature_set[indx1, :]
        batch_fg_feature_info = self.fg_feature_set[indx, :]
        # batch_bg_feature_info = self.bg_feature_set[indx2, :]
        batch_bg_feature_info = self.bg_feature_set[indx, :]

        batch_gt_feature_info = self.gt_feature_set[np.mod(indx, self.gt_num), :]
        # batch_gt_feature_info = self.gt_feature_set[indx_for_gt, :]

        batch_fg_feature = batch_fg_feature_info[:, 3:]
        batch_bg_feature = batch_bg_feature_info[:, 3:]
        batch_gt_feature = batch_gt_feature_info[:, 3:]

        # batch_feature_iou = np.concatenate((batch_fg_feature, batch_bg_feature), axis=0)
        # fg_feature_iou = batch_fg_feature

        batch_fg_iou = batch_fg_feature_info[:, 1].reshape(-1, 1)
        batch_bg_iou = batch_bg_feature_info[:, 1].reshape(-1, 1)

        # batch_iou_inf = np.concatenate((batch_fg_iou, batch_bg_iou), axis=0).reshape(-1, 1)
        # fg_iou_inf = batch_fg_iou.reshape(-1, 1)

        if self.dataset == 'coco':
            batch_label = self.seenclasses[batch_gt_feature_info[:, 2].astype(np.int16)].reshape(-1, )
        elif self.dataset == 'vg' or self.dataset == 'imagenet':
            batch_label = torch.from_numpy(batch_gt_feature_info[:, 2].astype(np.int16).reshape(-1, ))

        batch_att = self.attributes[batch_label, :]

        return torch.from_numpy(batch_gt_feature), batch_label, torch.from_numpy(batch_att), \
               torch.from_numpy(batch_fg_feature), torch.from_numpy(batch_fg_iou), \
               torch.from_numpy(batch_bg_feature), torch.from_numpy(batch_bg_iou)


        # self.pool = Pool(8)
        # if self.index.shape[0] < batch_size:
        #     self.index = np.random.permutation(range(self.ntrain))
        # indx = self.index[:batch_size].tolist()
        # self.index = self.index[batch_size:]
        # results = self.pool.map(load_feature, indx)
        # feature_batch = np.concatenate(results, axis=0)
        # self.pool.close()
        # self.pool.join()
        #
        # batch_feature = feature_batch[:, 3:]
        # batch_label = feature_batch[:, 2].astype(np.int16)
        # batch_att = self.attributes[batch_label, :]

        # batch_set = self.current_feature_set[:batch_size, :]
        # self.current_feature_set = self.current_feature_set[batch_size:, :]
        #
        # if self.current_feature_set.shape[0] < batch_size:
        #     self.file_number += 1
        #     self.feature_used += batch_size
        #     if self.feature_used > self.ntrain - 200:
        #         self.file_number = 1
        #     next_feature_set = self.load_feature_file(self.file_number)
        #     self.current_feature_set = np.concatenate((self.current_feature_set, next_feature_set), axis=0)
        #
        # batch_feature = batch_set[:, 1:]
        # batch_label = batch_set[:, 0].astype(np.int16)
        # batch_att = self.attributes[batch_label, :]
        #











