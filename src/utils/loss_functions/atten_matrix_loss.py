from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat

class selfsupervise_loss(nn.Module):

    def __init__(self, heads=6):
        super(selfsupervise_loss, self).__init__()
        self.heads = heads
        self.smoothl1 = torch.nn.SmoothL1Loss()

    def forward(self, attns, smooth=1e-40):
        layer = len(attns)
        for i in range(layer):
            attni = attns[i]  # b h n n
            b, h, n, d = attni.shape
            #attentionmap_visual(attni)
            # entropy loss
            log_attni = torch.log2(attni + smooth)
            entropy = -1 * torch.sum(attni * log_attni, dim=-1) / torch.log2(torch.tensor(n*1.0)) # b h n
            entropy_min = torch.min(entropy, dim=-1)[0]  # b h
            p_loss = (entropy_min-0.9).clamp_min(0)*(1/0.1)

            # symmetry loss
            attni_t = attni.permute(0, 1, 3, 2)
            #distance = torch.abs(attni_t*n - attni*n)  # b h n n
            #distance = torch.sum(distance, dim=-1)/n  # b h n
            #s_loss = torch.sum(distance, dim=-1)/n  # b h
            s_loss = self.smoothl1(attni*n, attni_t*n).clamp_min(0.1)

            if i == 0:
                loss = 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
            else:
                loss += 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)

        return loss / layer


class A_organs_loss(nn.Module):
    def __init__(self, depth_dict, patient_dict, heads=6):
        super(A_organs_loss, self).__init__()
        self.heads = heads
        self.smoothl1 = torch.nn.SmoothL1Loss()
        self.depth_dict   = depth_dict
        self.patient_dict = patient_dict


    def forward(self, attns, ids, A_organs, smooth=1e-40):
        # --------------- 将attns从layer b h n n的形式改成 b layer h n n，并对multi-layer multi-head进行求和求平均 ---------------
        layer = len(attns)
        b, h, n, d = attns[0].shape
        A_organs = A_organs.cuda()

        attns_tensor = torch.stack(attns)  # torch.Size([4, 8, 6, 1024, 1024]) cuda:0
        attns_mean_batch = torch.mean(torch.mean(attns_tensor, dim=0), dim=1)  # cuda:0, torch.Size([4, 8, 6, 1024, 1024]) -> torch.Size([8, 6, 1024, 1024]) -> torch.Size([8, 1024, 1024])
        # print('attns_tensor: ', attns_tensor.shape, attns_tensor.device)
        # print('attns_mean_batch: ', attns_mean_batch.shape, attns_mean_batch.device)

        loss = 0.0
        for j in range(b):
            # --------------------- obtain the corresponding A_organs label of the slice ---------------------
            slice_id = ids[j]
            patient_id = slice_id.split('.')[0].split('_')[0] + '_' + slice_id.split('.')[0].split('_')[1]
            # print('patient_id: ', patient_id, 'slice_id: ', slice_id)
            # print('self.patient_dict[patient_id]: ', self.patient_dict[patient_id])
            depth_location = self.patient_dict[patient_id].index(slice_id) + 1
            slice_depth_dict = self.depth_dict[patient_id]

            new_depth_loaction = round(depth_location * slice_depth_dict["server_depth_median"] / slice_depth_dict["original_depth"]) - 1

            # print('depth_location: ', depth_location, 'server_depth_median: ', slice_depth_dict["server_depth_median"], 'original_depth: ', slice_depth_dict["original_depth"], 'new_depth_loaction: ', new_depth_loaction)

            slice_A_organs = A_organs[new_depth_loaction]

            # --------------------- calculate loss ---------------------
            attnj = attns_mean_batch[j]  # n n
            assert slice_A_organs.shape == attnj.shape, 'slice_A_organs and attnj dimension mismatch, error, slice_A_organs shape is {}, attnj shape is {}'.format(slice_A_organs.shape, attnj.shape)
            # print('slice_A_organs: ', slice_A_organs.shape, type(slice_A_organs), slice_A_organs.device)
            # print('attnj: ', attnj.shape, type(attnj), attnj.device)
            loss += self.smoothl1(attnj, slice_A_organs)
        loss = loss / b

        return loss


class A_organs_IU_loss(nn.Module):
    def __init__(self, inter_alpha, inter_beta, union_alpha, union_beta):
        super(A_organs_IU_loss, self).__init__()
        self.inter_alpha, self.inter_beta, self.union_alpha, self.union_beta = inter_alpha, inter_beta, union_alpha, union_beta

    def forward(self, attns, attns_inter_label, attns_union_label, smooth=1e-40):

        n, _, _ = attns.shape
        attns_ = attns.view(n, -1)
        attns_inter_label_ = attns_inter_label.view(n, -1)
        attns_union_label_ = attns_union_label.view(n, -1)

        T_attns_             = 1 / (1 + torch.exp(- self.inter_alpha * (attns_ - self.inter_beta)))
        T_attns_union_label_ = 1 / (1 + torch.exp(- self.union_alpha * (attns_union_label_ - self.union_beta)))

        loss_inter = - (attns_inter_label_ * T_attns_).sum()             / (attns_inter_label_.sum() + smooth)
        loss_union = - (attns_             * T_attns_union_label_).sum() / (attns_.sum() + smooth)

        return loss_inter + loss_union


class A_organs_IUA_loss(nn.Module):
    def __init__(self, inter_alpha, union_alpha):
        super(A_organs_IUA_loss, self).__init__()
        self.inter_alpha, self.union_alpha = inter_alpha, union_alpha

    def forward(self, attns, attns_inter_label, attns_union_label, smooth=1e-40):

        n, h, w = attns.shape

        inter_beta = torch.mean(attns, dim=-1, keepdim=True)
        inter_beta = repeat(inter_beta, 'b h 1 -> b h w', w=w)
        inter_beta = inter_beta.contiguous()

        union_beta = torch.mean(attns_union_label, dim=-1, keepdim=True)
        union_beta = repeat(union_beta, 'b h 1 -> b h w', w=w)
        union_beta = union_beta.contiguous()

        attns_ = attns.view(n, -1)
        attns_inter_label_ = attns_inter_label.view(n, -1)
        attns_union_label_ = attns_union_label.view(n, -1)
        inter_beta_ = inter_beta.view(n, -1)
        union_beta_ = union_beta.view(n, -1)

        T_attns_             = 1 / (1 + torch.exp( - self.inter_alpha * (attns_ - inter_beta_)))
        T_attns_union_label_ = 1 / (1 + torch.exp( - self.union_alpha * (attns_union_label_ - union_beta_)))

        loss_inter = - (attns_inter_label_ * T_attns_).sum()             / (attns_inter_label_.sum() + smooth)
        loss_union = - (attns_             * T_attns_union_label_).sum() / (attns_.sum() + smooth)

        return loss_inter + loss_union


class A_organs_IUA_loss_v2(nn.Module):
    def __init__(self, inter_alpha, union_alpha):
        super(A_organs_IUA_loss_v2, self).__init__()
        self.inter_alpha, self.union_alpha = inter_alpha, union_alpha

    def forward(self, attns, attns_inter_label, attns_union_label, smooth=1e-40):

        n, h, w = attns.shape

        inter_beta = torch.mean(attns_inter_label, dim=-1, keepdim=True)
        inter_beta = repeat(inter_beta, 'b h 1 -> b h w', w=w)
        inter_beta = inter_beta.contiguous()

        union_beta = torch.mean(attns, dim=-1, keepdim=True)
        union_beta = repeat(union_beta, 'b h 1 -> b h w', w=w)
        union_beta = union_beta.contiguous()

        attns_ = attns.view(n, -1)
        attns_inter_label_ = attns_inter_label.view(n, -1)
        attns_union_label_ = attns_union_label.view(n, -1)
        inter_beta_ = inter_beta.view(n, -1)
        union_beta_ = union_beta.view(n, -1)

        T_attns_inter_label_ = 1 / (1 + torch.exp( - self.inter_alpha * (attns_inter_label_ - inter_beta_)))
        T_attns_             = 1 / (1 + torch.exp( - self.union_alpha * (attns_             - union_beta_)))

        loss_inter = (attns_inter_label_.sum() - (torch.min(attns_, attns_inter_label_) * T_attns_inter_label_).sum()) / (attns_inter_label_.sum() + smooth)
        loss_union = (attns_.sum()             - (torch.min(attns_, attns_union_label_) * T_attns_).sum())             / (attns_.sum() + smooth)

        return loss_inter + loss_union


class fea_organs_IU_loss(nn.Module):
    def __init__(self, inter_alpha, inter_beta, union_alpha, union_beta):
        super(fea_organs_IU_loss, self).__init__()
        self.inter_alpha, self.inter_beta, self.union_alpha, self.union_beta = inter_alpha, inter_beta, union_alpha, union_beta


    def forward(self, fea, fea_inter_label, fea_union_label, smooth=1e-40):

        n, _, _ = fea.shape
        fea_ = fea.view(n, -1)
        fea_inter_label_ = fea_inter_label.view(n, -1)
        fea_union_label_ = fea_union_label.view(n, -1)

        T_fea_             = 1 / (1 + torch.exp(- self.inter_alpha * (fea_ - self.inter_beta)))
        T_fea_union_label_ = 1 / (1 + torch.exp(- self.union_alpha * (fea_union_label_ - self.union_beta)))

        loss_inter = - (fea_inter_label_ * T_fea_).sum()             / (fea_inter_label_.sum() + smooth)
        loss_union = - (fea_             * T_fea_union_label_).sum() / (fea_.sum() + smooth)

        return loss_inter + loss_union


def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count



