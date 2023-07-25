import gc
import os
import logging
import torch
import itertools

import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from .utils import metrics
from .utils.loss_functions.atten_matrix_loss import selfsupervise_loss, A_organs_IU_loss, A_organs_IUA_loss
from collections import OrderedDict
from .utils.loss_functions.dice_loss import DC_and_CE_loss
from .utils.utils_gray import Dataset_Fed, JointTransform2D

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, writer, args, log_path):
        """Client object is initiated by the center server."""
        self.id            = client_id
        self.train_data    = local_data["train"]
        self.val_data      = local_data["val"]
        self.test_Fed_data = local_data["test"][0]
        self.test_BCV_data = local_data["test"][1]
        self.weight        = local_data["weight"].cuda()
        self.A_data        = local_data["A"]

        # self.device  = device
        self.__model  = None
        self.A_organs_inter = None
        self.A_organs_union = None
        self.writer  = writer
        self.args    = args
        self.log_path = log_path

        self.ssa_loss = selfsupervise_loss()
        self.DC_CE_loss = DC_and_CE_loss(soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, ce_kwargs={'weight': self.weight})
        # self.A_organs_IU_loss = A_organs_IU_loss(inter_alpha=self.args.inter_alpha, union_alpha=self.args.union_alpha, inter_beta=self.args.inter_beta, union_beta=self.args.union_beta)
        self.A_organs_IU_loss = A_organs_IUA_loss(inter_alpha=self.args.inter_alpha, union_alpha=self.args.union_alpha)


    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.train_data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.trainloader   = DataLoader(self.train_data,    batch_size=client_config["train_batch_size"], shuffle=True)
        self.valloader     = DataLoader(self.val_data,      batch_size=client_config["val_batch_size"],   shuffle=False)
        self.testFedloader = DataLoader(self.test_Fed_data, batch_size=client_config["test_batch_size"],  shuffle=False)
        self.testBCVloader = DataLoader(self.test_BCV_data, batch_size=client_config["test_batch_size"],  shuffle=False)

        self.Aloader = DataLoader(self.A_data, batch_size=client_config["train_batch_size"], shuffle=False)

        self.local_epoch  = client_config["num_local_epochs"]

        self.optimizer    = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

        self.depth_dict = self.create_depth_dict(server_depth_median=client_config["server_depth_median"])
        self.server_depth_median = client_config["server_depth_median"]

        # self.criterion     = client_config["criterion"]
        # self.A_organs_loss = A_organs_loss(depth_dict=self.depth_dict, patient_dict=self.train_data.patient_dict)

    def create_depth_dict(self, server_depth_median):
        '''

        :param server_depth_median:
        :return: {'liver_0001': {'original_depth': , 'client_depth_median': , 'server_depth_median': },
                  'liver_0002': {'original_depth': , 'client_depth_median': , 'server_depth_median': },
                  ......}
        '''
        depth_dict = OrderedDict()
        for id in self.train_data.ids:
            key = id.split('_')[0] + '_' + id.split('_')[1]
            if key not in depth_dict:
                depth_dict[key] = {'original_depth':len(self.train_data.patient_dict[key]), 'client_depth_median':self.train_data.depth_median, 'server_depth_median':server_depth_median}
        return depth_dict

    def client_update(self, idx, _round):
        """Update local model using local dataset."""
        self.model.train()
        self.model.cuda()

        assert idx == self.id, 'client error'

        loss_ss_train = 0.0

        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        for e in range(self.local_epoch):
            for data, labels, ids in self.trainloader:
                data, labels = data.float().cuda(), labels.long().cuda()  # torch.Size([8, 1, 256, 256]) torch.Size([8, 256, 256])

                optimizer.zero_grad()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]
                # print('train: ', idx, len(outputs), outputs[idx].shape, ids)
                # loss_en = eval(self.criterion)(self.weight)(outputs[idx], labels)
                loss_ed_ce = self.DC_CE_loss(outputs[idx], labels)
                loss_attn = self.ssa_loss(attns1)

                if _round <= self.args.begin_SS_round:
                    loss = self.args.en_weight * loss_ed_ce + self.args.attn_weight * loss_attn
                else:
                    attns_inter_label, attns_union_label = self.find_A_oragan_label(attns1, ids)  # torch.Size([8, 256, 256]) cuda:0 torch.Size([8, 256, 256]) cuda:0
                    attns = torch.mean(torch.mean(torch.stack(attns1), dim=0), dim=1)  # torch.Size([8, 256, 256]) cuda:0
                    loss_ss = self.A_organs_IU_loss(attns, attns_inter_label, attns_union_label)  # loss_ss:  tensor(-0.0025, device='cuda:0', grad_fn=<AddBackward0>)
                    loss_ss_train = loss_ss_train + loss_ss.item()
                    loss = self.args.en_weight * loss_ed_ce + self.args.attn_weight * loss_attn + self.args.ss_weight * loss_ss

                loss.backward()
                optimizer.step()

                # torch.cuda.empty_cache()
            #     break
            # break

        A_list_K, FI_list_K = self.obtain_dice_list(_round)
        # print('A_list_K: ', A_list_K)
        # print('FI_list_K: ', FI_list_K)

        A_organ_inter, A_organ_union, A_list_K_tmp = self.collect_A_oragan_IUK(_round, A_list_K)
        if _round > self.args.begin_FI_round:
            attr_every_head_record = self.calculate_filter_importance(FI_list_K)
        else:
            attr_every_head_record = None

        # print('attr_every_head_record: ', attr_every_head_record)

        loss_ss_train = loss_ss_train / (len(self.trainloader) * self.local_epoch)

        message = f"[Round: {str(_round).zfill(4)}] [Client {str(self.id).zfill(4)}] ...finished training!\
                            \n\t=> train loss ss: {loss_ss_train:.8f}\
                            \n\t=> {str(A_list_K_tmp)}"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.writer.add_scalar('Client_{}'.format(str(self.id).zfill(4)) + '_train_loss_ss', loss_ss_train, _round)

        self.model.to("cpu")

        return A_organ_inter, A_organ_union, attr_every_head_record

    def obtain_dice_list(self, _round):
        # time1 = time.time()

        patient_dict = self.train_data.patient_dict

        # -------------------------- initial A_dict and FI_dict --------------------------
        A_dict = OrderedDict()

        for key, value in patient_dict.items():
            A_dict[key]     = {'A':torch.zeros(size=(len(value), 256, 256)), 'tp':0.0, 'fp':0.0, 'tn':0.0, 'fn':0.0}

        # -------------------------- assign value to A_dict --------------------------
        for data, labels, ids in self.Aloader:
            # count += 1
            data, labels = data.float().cuda(), labels.long().cuda()  # torch.Size([8, 1, 256, 256]) torch.Size([8, 256, 256])

            # -------------------------- obtain self-attention map from model --------------------------
            # time3 = time.time()
            self.model.eval()
            self.model.cuda()
            with torch.no_grad():
                outputs = self.model(data)
                attns1 = outputs[4]  # attns1: 4 x torch.Size([8, 6, 256, 256]), layers x (b, head, n , n)
            # print('attns1: ', len(attns1), attns1[0].shape, attns1[1].shape, attns1[2].shape, attns1[3].shape)
            # break
            # time4 = time.time()
            # if count == 1:
            #     print('time4 - time3: ', time4 - time3)

            # --------------- multi-layer multi-head sum and average are applied to the attns ---------------
            # time5 = time.time()
            b, h, n, d = attns1[0].shape
            attns1_tensor = torch.stack(attns1)  #  torch.Size([4, 8, 6, 256, 256])
            attns_mean_batch = torch.mean(torch.mean(attns1_tensor, dim=0), dim=1)  # torch.Size([8, 4, 6, 256, 256]) -> torch.Size([8, 6, 256, 256]) -> torch.Size([8, 256, 256])
            # print('attns: ', attns1_tensor.shape, attns_mean_batch.shape, torch.mean(attns1_tensor, dim=0).shape, torch.mean(torch.mean(attns1_tensor, dim=0), dim=1).shape)


            # --------------- the quantization result of the segmentation result is obtained, that is, the values are 0,1,2,3 ---------------
            gt = labels.detach().cpu().numpy()
            y_out = F.softmax(outputs[self.id], dim=1)
            pred = y_out.detach().cpu().numpy()
            seg = np.argmax(pred, axis=1)  # b s h w -> b h w

            # --------------- assign attns_mean to A_dict, and get'tp', 'fp', 'tn', 'fn' ---------------
            # time7 = time.time()
            for j in range(b):
                slice_id   = ids[j]
                attns_mean = attns_mean_batch[j]  # (256, 256)
                patient_id = slice_id.split('.')[0].split('_')[0] + '_' + slice_id.split('.')[0].split('_')[1]
                # time11 = time.time()
                slice_location = patient_dict[patient_id].index(slice_id)
                # time12 = time.time()
                # if count == 1 and j == 1:
                #     print('time12 - time11: ', time12 - time11)
                A_dict[patient_id]['A'][slice_location] = attns_mean
                # print('slice_id: ', slice_id)
                # print('attns_mean: ', attns_mean.shape)
                # print('patient_id: ', patient_id)
                # print('patient_dict[patient_id]: ', patient_dict[patient_id])
                # print('slice_location: ', slice_location)

                seg_ = seg[j:j + 1]
                gt_  = gt[j:j + 1]
                tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                A_dict[patient_id]['tp'] += tp
                A_dict[patient_id]['fp'] += fp
                A_dict[patient_id]['tn'] += tn
                A_dict[patient_id]['fn'] += fn


        # -------------------------- the dice of each sample is calculated and then sorted to get the top 10% of the sample A_list_K --------------------------
        for key, content in A_dict.items():
            A_dict[key]['dice'] = 2 * content['tp'] / (2 * content['tp'] + content['fp'] + content['fn'] + 1e-40)
        A_list_K_tmp2 = sorted(A_dict.items(), key=lambda x: x[1]['dice'], reverse=True)
        A_list_K = list(itertools.islice(sorted(A_dict.items(), key=lambda x: x[1]['dice'], reverse=True), int(len(A_dict) * self.args.SSK)))  # descending order
        # print('A_list_K: ', A_list_K)  # [(4, {'A': 1, 'dice': 4}), (3, {'A': 1, 'dice': 3})]
        with open(os.path.join(self.log_path, 'sample_dice.txt'), 'a') as f:
            f.writelines('[Round: {}] [Client {}]'.format(str(_round).zfill(4), str(self.id).zfill(4)) + '\n')
            count = 0
            for element in A_list_K_tmp2:
                count += 1
                f.writelines(str(element[0]) + '   tp: ' + str(element[1]['tp']) + '   fp: ' + str(element[1]['fp']) + '   tn: ' + str(element[1]['tn']) + '   fn: ' + str(element[1]['fn']) + '   dice: ' + str(element[1]['dice']) + ' ' * 5)
                if count % 3 == 0:
                    f.writelines('\n')

        FI_list_K_tmp = list(itertools.islice(sorted(A_dict.items(), key=lambda x: x[1]['dice'], reverse=True), int(len(A_dict) * self.args.FIK)))
        FI_list_K = [element[0] for element in FI_list_K_tmp]

        return A_list_K, FI_list_K

    def collect_A_oragan_IUK(self, _round, A_list_K):

        server_depth_median = self.server_depth_median

        # -------------------------- modify depth od self attention map and obtain average self attention map --------------------------
        A_organ_inter = []
        A_organ_union = []
        A_list_K_tmp = []  # Responsible for printing out the selected sample
        count = 0
        for element in A_list_K:
            count += 1
            key = element[0]
            A   = element[1]['A']
            A_list_K_tmp.append({key:element[1]['dice']})
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
            # time11 = time.time()
            # new_A = scipy.ndimage.zoom(A, zoom=(scale, 1, 1))
            # if count == 1:
            #     print('A: ', A.contiguous().unsqueeze(0).unsqueeze(0).shape)
            new_A = F.interpolate(A.contiguous().unsqueeze(0).unsqueeze(0), size=(server_depth_median, A.shape[1], A.shape[2]), mode='trilinear')  # d h w  ->  1 1 d h w, torch.Size([1, 1, 41, 256, 256])
            # align_corners=True, False：https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/8
            # time12 = time.time()
            # if count == 1:
            #     print('time12 - time11: ', time12 - time11)
            # new_A_dict[key] = new_A
            A_organ_inter.append(new_A)
            A_organ_union.append(new_A)
            # print('key: ', key, 'A: ', A.shape, 'new_A: ', new_A.shape, 'original_depth: ', original_depth)

        A_organ_inter = torch.stack(A_organ_inter)  # torch.Size([7, 1, 1, 41, 256, 256])
        A_organ_union = torch.stack(A_organ_union)
        # print(A_organ_inter.shape, A_organ_union.shape)

        A_organ_inter = torch.min(A_organ_inter, dim=0).values  # torch.Size([1, 1, 41, 256, 256])
        A_organ_union = torch.max(A_organ_union, dim=0).values
        # print(A_organ_inter.shape, A_organ_union.shape)

        A_organ_inter = A_organ_inter.squeeze()  # torch.Size([41, 256, 256])    n 1 1 d h w  ->  1 1 d h w  ->  d h w
        A_organ_union = A_organ_union.squeeze()  #                                 n 1 1 d h w  ->  1 1 d h w  ->  d h w
        # print(A_organ_inter.shape, A_organ_union.shape)

        # print('A_organ_inter: ', A_organ_inter.shape, torch.stack(A_organ_inter).shape, (torch.min(torch.stack(A_organ_inter), dim=0)).values.shape)

        # time10 = time.time()
        # print('time10 - time9: ', time10 - time9)
        # print('time10 - time1: ', time10 - time1)

        return A_organ_inter, A_organ_union, A_list_K_tmp

    def find_A_oragan_label(self, attns, ids):

        with torch.no_grad():
            layer = len(attns)  # torch.Size([4, 8, 6, 256, 256]) cuda:0 -> torch.Size([8, 256, 256]) cuda:0
            b, h, n, d = attns[0].shape
            A_organs_depth = self.A_organs_inter.shape[0]

            attns_inter_label = torch.zeros(size=(b, n, d))  # torch.Size([8, 256, 256])
            attns_union_label = torch.zeros(size=(b, n, d))  # torch.Size([8, 256, 256])

            for j in range(b):
                # --------------------- obtain the corresponding A_organs label of the slice ---------------------
                slice_id = ids[j]
                patient_id = slice_id.split('.')[0].split('_')[0] + '_' + slice_id.split('.')[0].split('_')[1]
                # print('patient_id: ', patient_id, 'slice_id: ', slice_id)
                # print('self.patient_dict[patient_id]: ', self.patient_dict[patient_id])
                depth_location = self.train_data.patient_dict[patient_id].index(slice_id) + 1
                slice_depth_dict = self.depth_dict[patient_id]

                new_depth_loaction = round(depth_location * slice_depth_dict["server_depth_median"] / slice_depth_dict["original_depth"]) - 1

                # print('depth_location: ', depth_location, 'server_depth_median: ', slice_depth_dict["server_depth_median"], 'original_depth: ', slice_depth_dict["original_depth"], 'new_depth_loaction: ', new_depth_loaction)

                slice_A_organs_inter = (self.A_organs_inter[new_depth_loaction] + self.A_organs_inter[max(new_depth_loaction - 1, 0)]
                                        + self.A_organs_inter[min(new_depth_loaction + 1, A_organs_depth - 1)]) / 3
                slice_A_organs_union = (self.A_organs_union[new_depth_loaction] + self.A_organs_union[max(new_depth_loaction - 1, 0)]
                                        + self.A_organs_union[min(new_depth_loaction + 1, A_organs_depth - 1)]) / 3
                # print('new_depth_location: ', new_depth_loaction, max(new_depth_loaction - 1, 0), min(new_depth_loaction + 1, A_organs_depth - 1))

                attns_inter_label[j] = slice_A_organs_inter
                attns_union_label[j] = slice_A_organs_union

        # print('attns_inter_label: ', attns_inter_label.shape)
        return attns_inter_label.cuda(), attns_union_label.cuda()

    def scaled_input(self, emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
        # shape of emb: (batch_size, num_head, 256, 256)
        if baseline is None:
            baseline = torch.zeros_like(emb)

        num_points = num_batch
        scale = 1.0 / num_points
        if start_i is None:
            step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale  # (1, batch_size, num_head, 256, 256)
            res = torch.cat([torch.add(baseline.unsqueeze(0), step * i) for i in range(num_points)], dim=0)
            return res, step[0]
        else:
            step = (emb - baseline) * scale
            start_emb = torch.add(baseline, step * start_i)
            end_emb = torch.add(baseline, step * end_i)
            step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
            res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new * i) for i in range(num_points)], dim=0)
            return res, step_new[0]

    def calculate_filter_importance(self, FI_list_K):
        tf_val = JointTransform2D(img_size=256, crop=None, p_flip=0, p_gama=0, color_jitter_params=None, long_mask=True)
        if self.id == 0:
            FI_data = Dataset_Fed(dataset_path=self.args.data_path, label_value_scale=255, split=self.args.liver_train, joint_transform=tf_val, FI_condition_list=FI_list_K)
        elif self.id == 1:
            FI_data = Dataset_Fed(dataset_path=self.args.data_path, label_value_scale=255, split=self.args.kidney_train, joint_transform=tf_val, FI_condition_list=FI_list_K)
        elif self.id == 2:
            FI_data = Dataset_Fed(dataset_path=self.args.data_path, label_value_scale=255, split=self.args.pancreas_train, joint_transform=tf_val, FI_condition_list=FI_list_K)
        else:
            print('client id error')
            FI_data = None

        FI_loader = DataLoader(FI_data, batch_size=self.args.FI_batch_size, shuffle=False)

        attr_every_head_record = []
        num_layer = 4
        num_head = 6

        for data, labels, ids in FI_loader:

            data, labels = data.float().cuda(), labels.float().cuda()  # torch.Size([1, 1, 256, 256]) torch.Size([1, 256, 256])
            # id = ids[0]  # because batch size is 1

            self.model.eval()
            self.model.cuda()

            attr_every_head_record_for_one_batch = [0] * num_head * num_layer

            for tar_layer in range(num_layer):
                # print('tar_layer: ', tar_layer)
                with torch.no_grad():
                    out1, out2, out3, qkvs1, attns1, fea, tar_attn = self.model(x=data, tar_layer=tar_layer, tmp_score=None)
                pred_label = torch.argmax(torch.nn.functional.softmax(out3, dim=1), dim=1)

                baseline = None
                scale_att, step = self.scaled_input(emb=tar_attn.data, batch_size=self.args.FI_batch_size, num_batch=self.args.FI_num_batch, baseline=baseline, start_i=None, end_i=None)
                scale_att.requires_grad_(True)

                # print('tar_attn: ', tar_attn.shape)  # torch.Size([8, 6, 256, 256]), batch_size=8
                # print('scale_att: ', scale_att.shape)  # torch.Size([10, 8, 6, 256, 256]), num_batch=10
                # print('step: ', step.shape)  # torch.Size([8, 6, 256, 256]), batch_size=8
                # print('pred_label: ', pred_label.shape)  # torch.Size([8, 256, 256]), batch_size=8


                attr_all_for_one_layer = None

                for j_batch in range(self.args.FI_num_batch):
                    one_batch_att = scale_att[j_batch]
                    # print('one_batch_att: ', one_batch_att.shape)  # torch.Size([8, 6, 256, 256])
                    scale_out1, scale_out2, scale_out3, scale_qkvs1, scale_attns1, scale_fea, scale_tar_attn = self.model(
                        x=data,
                        tar_layer=tar_layer,
                        tmp_score=one_batch_att)
                    prob = torch.nn.functional.softmax(scale_out3, dim=1)  # torch.Size([8, 2, 256, 256])
                    gradient = torch.autograd.grad(prob.sum(), one_batch_att)  # The former takes the derivative of the latter, the numerator is the former, the denominator is the latter
                    # print('gradient: ', gradient, gradient[0].shape)  # torch.Size([8, 6, 256, 256])
                    grad = gradient[0].sum(dim=0)  # sum on batc size dimension
                    # print('grad: ', grad.shape)  # torch.Size([6, 256, 256])
                    attr_all_for_one_layer = grad if attr_all_for_one_layer is None else torch.add(attr_all_for_one_layer, grad)

                    # break

                attr_all_for_one_layer = attr_all_for_one_layer * step
                # print('attr_all: ', attr_all_for_one_layer.shape)  # torch.Size([8, 6, 256, 256])

                for i_batch in range(data.shape[0]):
                    attr_all_for_one_layer_batch = attr_all_for_one_layer[i_batch]  # torch.Size([6, 256, 256])
                    for i_head in range(0, num_head):
                        attr_every_head_record_for_one_batch[tar_layer * num_head + i_head] += float(
                            attr_all_for_one_layer_batch[i_head].max())

                # break
            # print(ids, attr_every_head_record_for_one_batch)
            # print(ids)

            attr_every_head_record.append(np.array(attr_every_head_record_for_one_batch))
            # break

        attr_every_head_record = np.array(attr_every_head_record)
        attr_every_head_record = attr_every_head_record.sum(axis=0)
        self.model.zero_grad()
        return attr_every_head_record

    def client_evaluate(self, idx, _round):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.cuda()

        assert idx == self.id, 'client error'

        val_loss = 0

        patientnumber = 450  # record the patients
        flag = np.zeros(patientnumber)  # record the patients
        tps, fps = np.zeros(patientnumber), np.zeros(patientnumber)
        tns, fns = np.zeros(patientnumber), np.zeros(patientnumber)

        with torch.no_grad():
            for data, labels, ids in self.valloader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]
                # loss_en = eval(self.criterion)(self.weight)(outputs[idx], labels)
                loss_ed_ce = self.DC_CE_loss(outputs[idx], labels)
                loss_attn = self.ssa_loss(attns1)
                val_loss = val_loss + loss_ed_ce.item() + loss_attn.item()


                # val_loss = val_loss + loss_en.item() + loss_attn.item()

                gt    = labels.detach().cpu().numpy()
                y_out = F.softmax(outputs[idx], dim=1)
                pred  = y_out.detach().cpu().numpy()
                seg   = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_  = ids[b_id]
                    seg_ = seg[b_id:b_id+1]
                    gt_  = gt[b_id:b_id+1]

                    if idx == int(0):
                        patientid = int(id_[6:10])
                    elif idx == int(1):
                        patientid = int(id_[7:11])
                    elif idx == int(2):
                        patientid = int(id_[9:13])
                    else:
                        print('eval error')

                    if flag[patientid] != 1:
                        flag[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps[patientid] += tp
                    fps[patientid] += fp
                    tns[patientid] += tn
                    fns[patientid] += fn

                    # torch.cuda.empty_cache()
                # break


        # ---------------------------- calculate total dice ----------------------------
        patients = np.sum(flag)
        tps = tps[flag > 0]
        fps = fps[flag > 0]
        tns = tns[flag > 0]
        fns = fns[flag > 0]
        dice = 2 * tps / (2 * tps + fps + fns + 1e-40)  # p c
        mdice = np.mean(dice, axis=0)  # c

        self.model.to("cpu")

        val_loss = val_loss / len(self.valloader)
        # print('val loss: ', val_loss)

        message = f"[Round: {str(_round).zfill(4)}] [Client {str(self.id).zfill(4)}] ...finished evaluation!\
                    \t=> val loss: {val_loss:.4f}, dice: {mdice:.4f}"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.writer.add_scalar('Client_{}'.format(str(self.id).zfill(4)) + '_val_loss', val_loss, _round)
        self.writer.add_scalar('Client_{}'.format(str(self.id).zfill(4)) + '_val_dices', mdice, _round)

        return val_loss, mdice
