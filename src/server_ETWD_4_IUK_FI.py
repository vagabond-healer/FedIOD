import copy

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .models.ETWD_4 import *
from .client_ETWD_4_IUK_FI import Client
from .utils.utils_gray import *
from .utils.utils_inital import *
from .utils import metrics
import torch.nn.functional
from .utils.loss_functions.atten_matrix_loss import selfsupervise_loss

logger = logging.getLogger(__name__)


class Server(object):

    def __init__(self, writer, args, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}, log_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer
        self.args = args

        self.model = eval(model_config["name"])(**model_config)
        if self.args.load_trained_model:
            self.model.load_state_dict(torch.load(self.args.load_model_path))

        self.seed = global_config["seed"]
        self.mp_flag = global_config["is_mp"]

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.train_batch_size = data_config["batch_size"][0]
        self.val_batch_size = data_config["batch_size"][1]
        self.test_batch_size = data_config["batch_size"][2]
        self.full_label_scale = data_config["full_label_scale"]
        self.partial_label_scale = data_config["partial_label_scale"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.optimizer = fed_config["optimizer"]

        self.optim_config = optim_config

        self.log_path = log_config["log_path"]
        self.checkpoints_path = log_config["model_save_path"]
        self.checkpoints_tmp_path = log_config["model_tmp_save_path"]
        self.checkpoints_BCV_path = log_config["model_BCV_save_path"]


        self.criterion = fed_config["criterion"]
        self.ssa_loss = selfsupervise_loss()

        self.tf_train = JointTransform2D(img_size=256, crop=None, p_flip=0.0, p_rota=0.5, p_scale=0.0, p_gaussn=0.0, p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
        self.tf_val = JointTransform2D(img_size=256, crop=None, p_flip=0, p_gama=0, color_jitter_params=None, long_mask=True)  # image reprocessing

    def setup(self, args, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        # init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # create dataset(train, val, test, weight, A) for each client
        datasets = create_datasets(data_path=self.data_path, datasets_name=self.dataset_name,
                                   partial_label_value_scale=self.partial_label_scale,
                                   full_label_value_scale=self.full_label_scale, tf_train=self.tf_train,
                                   tf_val=self.tf_val, args=args)  # [liver_dict, kidney_dict, pancreas_dict]

        # pancreas_train = datasets[2]['train']
        # print('pancreas_train.ids: ', pancreas_train.ids)
        # print('pancreas_train.patient_dict: ', pancreas_train.patient_dict)

        # calculate depth median and initial attention map
        self.liver_depth_median = datasets[0]["train"].depth_median  # 47
        self.kidney_depth_median = datasets[1]["train"].depth_median  # 40
        self.pancreas_depth_median = datasets[2]["train"].depth_median  # 23
        self.server_depth_median = round(np.median(
            np.array([self.liver_depth_median, self.kidney_depth_median, self.pancreas_depth_median])).item())  # 40
        self.A_organs_inter = torch.zeros([self.server_depth_median, 256, 256])
        self.A_organs_union = torch.zeros([self.server_depth_median, 256, 256])

        # prepare hold-out dataset for evaluation
        self.liver_val_data = datasets[0]["val"]  # load liver validation dataset
        self.liver_val_loader = DataLoader(self.liver_val_data, batch_size=self.val_batch_size, shuffle=False)
        self.liver_test_data = datasets[0]["test"][0]  # load liver test dataset
        self.liver_test_loader = DataLoader(self.liver_test_data, batch_size=self.test_batch_size, shuffle=False)

        self.kidney_val_data = datasets[1]["val"]  # load kidney validation dataset
        self.kidney_val_loader = DataLoader(self.kidney_val_data, batch_size=self.val_batch_size, shuffle=False)
        self.kidney_test_data = datasets[1]["test"][0]  # load kidney test dataset
        self.kidney_test_loader = DataLoader(self.kidney_test_data, batch_size=self.test_batch_size, shuffle=False)

        self.pancreas_val_data = datasets[2]["val"]  # load pancreas validation dataset
        self.pancreas_val_loader = DataLoader(self.pancreas_val_data, batch_size=self.val_batch_size, shuffle=False)
        self.pancreas_test_data = datasets[2]["test"][0]  # load pancreas test dataset
        self.pancreas_test_loader = DataLoader(self.pancreas_test_data, batch_size=self.test_batch_size, shuffle=False)

        self.BCV_test_data = datasets[0]["test"][1]  #   # load BCV dataset
        self.BCV_test_loader = DataLoader(self.BCV_test_data, batch_size=self.test_batch_size, shuffle=False)

        # assign dataset to each client
        self.clients = self.create_clients(datasets, args)

        # configure detailed settings for client upate and
        self.setup_clients(
            train_batch_size=self.train_batch_size, val_batch_size=self.val_batch_size,
            test_batch_size=self.test_batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config,
            server_depth_median=self.server_depth_median
        )


        # send the model skeleton and attention map to all clients
        self.transmit_model()

    def create_clients(self, local_datasets, args):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, writer=self.writer, args=args, log_path=self.log_path)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            if self.args.load_trained_model:
                A_organs_inter = torch.zeros((self.server_depth_median, 256, 256))
                A_organs_union = torch.zeros((self.server_depth_median, 256, 256))
                for client in tqdm(self.clients, leave=False):
                    local_A_organ_inter, local_A_organ_union = client.collect_A_oragan_IUK()
                    A_organs_inter += local_A_organ_inter
                    A_organs_union += local_A_organ_union
                A_organs_inter_avg = A_organs_inter / 3
                A_organs_union_avg = A_organs_union / 3

                # calculate self-attention matrix weight and global self-attention matrix
                A_organs_inter_bias = []
                A_organs_union_bias = []
                for client in tqdm(self.clients, leave=False):
                    local_A_organ_inter, local_A_organ_union = client.collect_A_oragan_IUK()
                    A_organs_inter_bias.append((local_A_organ_inter - A_organs_inter_avg) ** 2)
                    A_organs_union_bias.append((local_A_organ_union - A_organs_union_avg) ** 2)
                A_organs_inter_bias = torch.stack(A_organs_inter_bias)  # torch.Size([3, 41, 256, 256])
                A_organs_union_bias = torch.stack(A_organs_union_bias)  # torch.Size([3, 41, 256, 256])

                # print('A_organs_inter_bias: ', A_organs_inter_bias.shape)
                # print('A_organs_union_bias: ', A_organs_union_bias.shape)

                A_organs_inter_weight = torch.softmax(- A_organs_inter_bias * self.args.scale, dim=0)
                A_organs_union_weight = torch.softmax(- A_organs_union_bias * self.args.scale, dim=0)
                self.A_organs_inter = self.clients[0].collect_A_oragan_IUK()[0] * A_organs_inter_weight[0] \
                                      + self.clients[1].collect_A_oragan_IUK()[0] * A_organs_inter_weight[1] \
                                      + self.clients[2].collect_A_oragan_IUK()[0] * A_organs_inter_weight[2]  # torch.Size([41, 256, 256])
                self.A_organs_union = self.clients[0].collect_A_oragan_IUK()[1] * A_organs_union_weight[0] \
                                      + self.clients[1].collect_A_oragan_IUK()[1] * A_organs_union_weight[1] \
                                      + self.clients[2].collect_A_oragan_IUK()[1] * A_organs_union_weight[2]  # torch.Size([41, 256, 256])

            for client in tqdm(self.clients, leave=False):
                client.A_organs_inter = copy.deepcopy(self.A_organs_inter)
                client.A_organs_union = copy.deepcopy(self.A_organs_union)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
                self.clients[idx].A_organs_inter = copy.deepcopy(self.A_organs_inter)
                self.clients[idx].A_organs_union = copy.deepcopy(self.A_organs_union)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices)), sampled_client_indices} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices

    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {sampled_client_indices} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        A_organs_dict = OrderedDict()
        FI_heads_dict = OrderedDict()
        for idx in tqdm(sampled_client_indices, leave=False):
            A_organ_inter, A_organ_union, attr_every_head_record = self.clients[idx].client_update(idx, self._round)
            selected_total_size += len(self.clients[idx])
            A_organs_dict[str(idx)] = {'inter':A_organ_inter, 'union':A_organ_union}
            FI_heads_dict[str(idx)] = attr_every_head_record
            # print('attr_every_head_record: ', attr_every_head_record, len(attr_every_head_record))

            message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[idx].id).zfill(4)} is selected and updated (with total sample size: {str(len(self.clients[idx]))})!"
            print(message); logging.info(message)
            del message; gc.collect()
            # break

        return selected_total_size, A_organs_dict, FI_heads_dict

    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update(selected_index, self._round)
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def ETWD_IUK_FI_average_model(self, sampled_client_indices, original_mixing_coefficients, A_organs_dict, FI_heads_dict):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        # -------------- norm depth on local clients, then update to server --------------
        A_organs_inter = torch.zeros((self.server_depth_median, 256, 256))
        A_organs_union = torch.zeros((self.server_depth_median, 256, 256))
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_A_organ_inter = A_organs_dict[str(idx)]['inter']
            local_A_organ_union = A_organs_dict[str(idx)]['union']
            A_organs_inter += local_A_organ_inter
            A_organs_union += local_A_organ_union
        A_organs_inter_avg = A_organs_inter / 3
        A_organs_union_avg = A_organs_union / 3


        # calculate self-attention matrix weight and global self-attention matrix
        A_organs_inter_bias = []
        A_organs_union_bias = []
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_A_organ_inter = A_organs_dict[str(idx)]['inter']
            local_A_organ_union = A_organs_dict[str(idx)]['union']
            A_organs_inter_bias.append((local_A_organ_inter - A_organs_inter_avg) ** 2)
            A_organs_union_bias.append((local_A_organ_union - A_organs_union_avg) ** 2)


        A_organs_inter_bias = torch.stack(A_organs_inter_bias)  # torch.Size([3, 41, 256, 256])
        A_organs_union_bias = torch.stack(A_organs_union_bias)  # torch.Size([3, 41, 256, 256])

        # print('A_organs_inter_bias: ', A_organs_inter_bias.shape)
        # print('A_organs_union_bias: ', A_organs_union_bias.shape)

        A_organs_inter_weight = torch.softmax(- A_organs_inter_bias * self.args.scale_square, dim=0)
        A_organs_union_weight = torch.softmax(- A_organs_union_bias * self.args.scale_square, dim=0)


        self.A_organs_inter = A_organs_dict[str(0)]['inter'] * A_organs_inter_weight[0] \
                              + A_organs_dict[str(1)]['inter'] * A_organs_inter_weight[1] \
                              + A_organs_dict[str(2)]['inter'] * A_organs_inter_weight[2]   # torch.Size([41, 256, 256])
        self.A_organs_union = A_organs_dict[str(0)]['union'] * A_organs_union_weight[0] \
                              + A_organs_dict[str(1)]['union'] * A_organs_union_weight[1] \
                              + A_organs_dict[str(2)]['union'] * A_organs_union_weight[2]   # torch.Size([41, 256, 256])

        # calculate averaging coefficient of weights
        # print('round: ', round)
        if self._round > self.args.begin_FI_round:

            attr_every_head_record = torch.zeros([3, 24])
            attr_every_head_record[0] = torch.from_numpy(np.array(FI_heads_dict['0']))
            attr_every_head_record[1] = torch.from_numpy(np.array(FI_heads_dict['1']))
            attr_every_head_record[2] = torch.from_numpy(np.array(FI_heads_dict['2']))
            attr_every_head_record = torch.softmax(attr_every_head_record, dim=0)
            attr_every_head_record = attr_every_head_record.numpy()

            message = f"[Round: {str(self._round).zfill(4)}]  liver: {str(FI_heads_dict['0'])}, kidney: {str(FI_heads_dict['1'])}, pancreas: {str(FI_heads_dict['2'])}, attr_every_head_record: {str(attr_every_head_record)}"
            print(message); logging.info(message)
            del message; gc.collect()

            # average all clients parameters
            averaged_weights = OrderedDict()
            mixing_coefficients = original_mixing_coefficients
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                loacal_attr_every_head_record = attr_every_head_record[idx]
                for key in self.model.state_dict().keys():
                    # print('key: ', key)
                    if key.startswith('up1'):
                        if idx == 0:
                            averaged_weights[key] = local_weights[key]
                        else:
                            pass
                    elif key.startswith('up2'):
                        if idx == 1:
                            averaged_weights[key] = local_weights[key]
                        else:
                            pass
                    elif key.startswith('up3'):
                        if idx == 2:
                            averaged_weights[key] = local_weights[key]
                        else:
                            pass
                    elif key.startswith('trans4.transformer.layers.0.0.fn.to_qkv.weight'):
                        layer_id = 0
                        head_num = 6
                        layer_head_attr = torch.tensor(loacal_attr_every_head_record[layer_id * head_num: (layer_id + 1) * head_num])  # torch.Size([6])
                        layer_head_attr = repeat(layer_head_attr, 'head_num -> 256 (3 head_num 256)')  # torch.Size([256, 4608])
                        layer_head_attr_T = layer_head_attr.permute(1, 0)  # torch.Size([4608, 256])
                        if it == 0:
                            averaged_weights[key] = layer_head_attr_T * local_weights[key]
                        else:
                            averaged_weights[key] += layer_head_attr_T * local_weights[key]

                    elif key.startswith('trans4.transformer.layers.1.0.fn.to_qkv.weight'):
                        layer_id = 1
                        head_num = 6
                        layer_head_attr = torch.tensor(loacal_attr_every_head_record[layer_id * head_num: (
                                                                                                                      layer_id + 1) * head_num])  # torch.Size([6])
                        layer_head_attr = repeat(layer_head_attr, 'head_num -> 256 (3 head_num 256)')  # torch.Size([256, 4608])
                        layer_head_attr_T = layer_head_attr.permute(1, 0)  # torch.Size([4608, 256])
                        if it == 0:
                            averaged_weights[key] = layer_head_attr_T * local_weights[key]
                        else:
                            averaged_weights[key] += layer_head_attr_T * local_weights[key]
                    elif key.startswith('trans4.transformer.layers.2.0.fn.to_qkv.weight'):
                        layer_id = 2
                        head_num = 6
                        layer_head_attr = torch.tensor(loacal_attr_every_head_record[layer_id * head_num: (
                                                                                                                      layer_id + 1) * head_num])  # torch.Size([6])
                        layer_head_attr = repeat(layer_head_attr, 'head_num -> 256 (3 head_num 256)')  # torch.Size([256, 4608])
                        layer_head_attr_T = layer_head_attr.permute(1, 0)  # torch.Size([4608, 256])
                        if it == 0:
                            averaged_weights[key] = layer_head_attr_T * local_weights[key]
                        else:
                            averaged_weights[key] += layer_head_attr_T * local_weights[key]
                    elif key.startswith('trans4.transformer.layers.3.0.fn.to_qkv.weight'):
                        layer_id = 3
                        head_num = 6
                        layer_head_attr = torch.tensor(loacal_attr_every_head_record[layer_id * head_num: (
                                                                                                                      layer_id + 1) * head_num])  # torch.Size([6])
                        layer_head_attr = repeat(layer_head_attr, 'head_num -> 256 (3 head_num 256)')  # torch.Size([256, 4608])
                        layer_head_attr_T = layer_head_attr.permute(1, 0)  # torch.Size([4608, 256])
                        if it == 0:
                            averaged_weights[key] = layer_head_attr_T * local_weights[key]
                        else:
                            averaged_weights[key] += layer_head_attr_T * local_weights[key]
                        # print(key, local_weights[key].shape)
                        # '''
                        # trans4.transformer.layers.0.0.norm.weight torch.Size([256])
                        # trans4.transformer.layers.0.0.norm.bias torch.Size([256])
                        # trans4.transformer.layers.0.0.fn.headsita torch.Size([6])
                        # trans4.transformer.layers.0.0.fn.relative_position_bias_table torch.Size([961, 6])
                        # trans4.transformer.layers.0.0.fn.to_qkv.weight torch.Size([4608, 256])
                        # trans4.transformer.layers.0.0.fn.to_out.0.weight torch.Size([256, 1536])
                        # trans4.transformer.layers.0.0.fn.to_out.0.bias torch.Size([256])
                        # trans4.transformer.layers.0.1.norm.weight torch.Size([256])
                        # trans4.transformer.layers.0.1.norm.bias torch.Size([256])
                        # trans4.transformer.layers.0.1.fn.net.0.weight torch.Size([1024, 256])
                        # trans4.transformer.layers.0.1.fn.net.0.bias torch.Size([1024])
                        # trans4.transformer.layers.0.1.fn.net.3.weight torch.Size([256, 1024])
                        # trans4.transformer.layers.0.1.fn.net.3.bias torch.Size([256])
                        #
                        # trans4.transformer.layers.1.0.norm.weight
                        # trans4.transformer.layers.1.0.norm.bias
                        # trans4.transformer.layers.1.0.fn.headsita
                        # trans4.transformer.layers.1.0.fn.relative_position_bias_table
                        # trans4.transformer.layers.1.0.fn.to_qkv.weight
                        # trans4.transformer.layers.1.0.fn.to_out.0.weight
                        # trans4.transformer.layers.1.0.fn.to_out.0.bias
                        # trans4.transformer.layers.1.1.norm.weight
                        # trans4.transformer.layers.1.1.norm.bias
                        # trans4.transformer.layers.1.1.fn.net.0.weight
                        # trans4.transformer.layers.1.1.fn.net.0.bias
                        # trans4.transformer.layers.1.1.fn.net.3.weight
                        # trans4.transformer.layers.1.1.fn.net.3.bias
                        # trans4.transformer.layers.2.0.norm.weight
                        # trans4.transformer.layers.2.0.norm.bias
                        # trans4.transformer.layers.2.0.fn.headsita
                        # trans4.transformer.layers.2.0.fn.relative_position_bias_table
                        # trans4.transformer.layers.2.0.fn.to_qkv.weight
                        # trans4.transformer.layers.2.0.fn.to_out.0.weight
                        # trans4.transformer.layers.2.0.fn.to_out.0.bias
                        # trans4.transformer.layers.2.1.norm.weight
                        # trans4.transformer.layers.2.1.norm.bias
                        # trans4.transformer.layers.2.1.fn.net.0.weight
                        # trans4.transformer.layers.2.1.fn.net.0.bias
                        # trans4.transformer.layers.2.1.fn.net.3.weight
                        # trans4.transformer.layers.2.1.fn.net.3.bias
                        # trans4.transformer.layers.3.0.norm.weight
                        # trans4.transformer.layers.3.0.norm.bias
                        # trans4.transformer.layers.3.0.fn.headsita
                        # trans4.transformer.layers.3.0.fn.relative_position_bias_table
                        # trans4.transformer.layers.3.0.fn.to_qkv.weight
                        # trans4.transformer.layers.3.0.fn.to_out.0.weight
                        # trans4.transformer.layers.3.0.fn.to_out.0.bias
                        # trans4.transformer.layers.3.1.norm.weight
                        # trans4.transformer.layers.3.1.norm.bias
                        # trans4.transformer.layers.3.1.fn.net.0.weight
                        # trans4.transformer.layers.3.1.fn.net.0.bias
                        # trans4.transformer.layers.3.1.fn.net.3.weight
                        # trans4.transformer.layers.3.1.fn.net.3.bias
                        #
                        # '''

                    else:
                        if it == 0:
                            averaged_weights[key] = mixing_coefficients[it] * local_weights[key]
                        else:
                            averaged_weights[key] += mixing_coefficients[it] * local_weights[key]
            self.model.load_state_dict(averaged_weights)

        else:
            mixing_coefficients = original_mixing_coefficients

            # average all clients parameters
            averaged_weights = OrderedDict()
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if key.startswith('up1'):
                        if idx == 0:
                            averaged_weights[key] = local_weights[key]
                        else:
                            pass
                    elif key.startswith('up2'):
                        if idx == 1:
                            averaged_weights[key] = local_weights[key]
                        else:
                            pass
                    elif key.startswith('up3'):
                        if idx == 2:
                            averaged_weights[key] = local_weights[key]
                        else:
                            pass
                    else:
                        if it == 0:
                            averaged_weights[key] = mixing_coefficients[it] * local_weights[key]
                        else:
                            averaged_weights[key] += mixing_coefficients[it] * local_weights[key]
            self.model.load_state_dict(averaged_weights)



        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!\\n\t mixing_coefficients: {str(mixing_coefficients)}\n"
        print(message); logging.info(message)
        del message; gc.collect()

    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate(idx, self._round)
            message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(self.clients[idx].id).zfill(4)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_evaluate(selected_index, self._round)

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of selected clients {str(self.clients[selected_index].id).zfill(4)}!"
        print(message); logging.info(message)
        del message; gc.collect()

        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()  # [0, 1, 2]

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size, A_organs_dict, FI_heads_dict = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        original_mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]  # [0.2797413793103448, 0.37981974921630096, 0.3404388714733542]

        # original_mixing_coefficients = [0.2797413793103448, 0.37981974921630096, 0.3404388714733542]

        # average each updated model parameters of the selected clients and update the global model
        self.ETWD_IUK_FI_average_model(sampled_client_indices=sampled_client_indices, original_mixing_coefficients=original_mixing_coefficients, A_organs_dict=A_organs_dict, FI_heads_dict=FI_heads_dict)

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.cuda()

        patientnumber = 450

        val_liver_loss = 0
        flag_liver = np.zeros(patientnumber)  # record the patients
        tps_liver, fps_liver = np.zeros(patientnumber), np.zeros(patientnumber)
        tns_liver, fns_liver = np.zeros(patientnumber), np.zeros(patientnumber)

        val_kidney_loss = 0
        flag_kidney = np.zeros(patientnumber)  # record the patients
        tps_kidney, fps_kidney = np.zeros(patientnumber), np.zeros(patientnumber)
        tns_kidney, fns_kidney = np.zeros(patientnumber), np.zeros(patientnumber)

        val_pancreas_loss = 0
        flag_pancreas = np.zeros(patientnumber)  # record the patients
        tps_pancreas, fps_pancreas = np.zeros(patientnumber), np.zeros(patientnumber)
        tns_pancreas, fns_pancreas = np.zeros(patientnumber), np.zeros(patientnumber)

        with torch.no_grad():
            # -------------------------------------------- liver ---------------------------------------------------
            for data, labels, ids in self.liver_val_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]
                loss_en = eval(self.criterion)()(outputs[0], labels)
                loss_attn = self.ssa_loss(attns1)
                val_liver_loss = val_liver_loss + loss_en.item() + loss_attn.item()

                gt = labels.detach().cpu().numpy()
                y_out = torch.nn.functional.softmax(outputs[0], dim=1)
                pred = y_out.detach().cpu().numpy()
                seg = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    seg_ = seg[b_id:b_id + 1]
                    gt_ = gt[b_id:b_id + 1]

                    patientid = int(id_[6:10])  # liver_0001_1
                    if flag_liver[patientid] != 1:
                        flag_liver[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps_liver[patientid] += tp
                    fps_liver[patientid] += fp
                    tns_liver[patientid] += tn
                    fns_liver[patientid] += fn

                # torch.cuda.empty_cache()
                # break

            #  ---------------------------- calculate total liver loss and dice ----------------------------
            val_liver_loss = val_liver_loss / len(self.liver_val_loader)

            patients_liver = np.sum(flag_liver)
            tps_liver = tps_liver[flag_liver > 0]
            fps_liver = fps_liver[flag_liver > 0]
            tns_liver = tns_liver[flag_liver > 0]
            fns_liver = fns_liver[flag_liver > 0]
            dice_liver = 2 * tps_liver / (2 * tps_liver + fps_liver + fns_liver + 1e-40)  # p c
            mdice_liver = np.mean(dice_liver, axis=0)  # c

            # -------------------------------------------- kidney ---------------------------------------------------
            for data, labels, ids in self.kidney_val_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]
                loss_en = eval(self.criterion)()(outputs[1], labels)
                loss_attn = self.ssa_loss(attns1)
                val_kidney_loss = val_kidney_loss + loss_en.item() + loss_attn.item()

                gt = labels.detach().cpu().numpy()
                y_out = torch.nn.functional.softmax(outputs[1], dim=1)
                pred = y_out.detach().cpu().numpy()
                seg = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    seg_ = seg[b_id:b_id + 1]
                    gt_ = gt[b_id:b_id + 1]

                    patientid = int(id_[7:11])  # kidney_0001_1
                    if flag_kidney[patientid] != 1:
                        flag_kidney[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps_kidney[patientid] += tp
                    fps_kidney[patientid] += fp
                    tns_kidney[patientid] += tn
                    fns_kidney[patientid] += fn

                # torch.cuda.empty_cache()
                # break

            #  ---------------------------- calculate total kidney loss and dice ----------------------------
            val_kidney_loss = val_kidney_loss / len(self.kidney_val_loader)
            patients_kidney = np.sum(flag_kidney)
            tps_kidney = tps_kidney[flag_kidney > 0]
            fps_kidney = fps_kidney[flag_kidney > 0]
            tns_kidney = tns_kidney[flag_kidney > 0]
            fns_kidney = fns_kidney[flag_kidney > 0]
            dice_kidney = 2 * tps_kidney / (2 * tps_kidney + fps_kidney + fns_kidney + 1e-40)  # p c
            mdice_kidney = np.mean(dice_kidney, axis=0)  # c

            # -------------------------------------------- pancreas ---------------------------------------------------
            for data, labels, ids in self.pancreas_val_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]
                loss_en = eval(self.criterion)()(outputs[2], labels)
                loss_attn = self.ssa_loss(attns1)
                val_pancreas_loss = val_pancreas_loss + loss_en.item() + loss_attn.item()

                gt = labels.detach().cpu().numpy()
                y_out = torch.nn.functional.softmax(outputs[2], dim=1)
                pred = y_out.detach().cpu().numpy()
                seg = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    seg_ = seg[b_id:b_id + 1]
                    gt_ = gt[b_id:b_id + 1]

                    patientid = int(id_[9:13])  # pancreas_0001_1
                    if flag_pancreas[patientid] != 1:
                        flag_pancreas[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps_pancreas[patientid] += tp
                    fps_pancreas[patientid] += fp
                    tns_pancreas[patientid] += tn
                    fns_pancreas[patientid] += fn

                # torch.cuda.empty_cache()
                # break

            #  ---------------------------- calculate total pancreas loss and dice ----------------------------
            val_pancreas_loss = val_pancreas_loss / len(self.pancreas_val_loader)

            patients_pancreas = np.sum(flag_pancreas)
            tps_pancreas = tps_pancreas[flag_pancreas > 0]
            fps_pancreas = fps_pancreas[flag_pancreas > 0]
            tns_pancreas = tns_pancreas[flag_pancreas > 0]
            fns_pancreas = fns_pancreas[flag_pancreas > 0]
            dice_pancreas = 2 * tps_pancreas / (2 * tps_pancreas + fps_pancreas + fns_pancreas + 1e-40)  # p c
            mdice_pancreas = np.mean(dice_pancreas, axis=0)  # c

        self.model.to("cpu")

        val_loss = val_liver_loss + val_kidney_loss + val_pancreas_loss
        dice = (mdice_liver + mdice_kidney + mdice_pancreas) / 3
        return val_loss, dice, mdice_liver, mdice_kidney, mdice_pancreas

    def test_global_model_on_Fed(self):
        """Test the global model using Fed and BCV."""
        self.model.eval()
        self.model.cuda()

        patientnumber = 450

        flag_liver = np.zeros(patientnumber)  # record the patients
        tps_liver, fps_liver = np.zeros(patientnumber), np.zeros(patientnumber)
        tns_liver, fns_liver = np.zeros(patientnumber), np.zeros(patientnumber)

        flag_kidney = np.zeros(patientnumber)  # record the patients
        tps_kidney, fps_kidney = np.zeros(patientnumber), np.zeros(patientnumber)
        tns_kidney, fns_kidney = np.zeros(patientnumber), np.zeros(patientnumber)

        flag_pancreas = np.zeros(patientnumber)  # record the patients
        tps_pancreas, fps_pancreas = np.zeros(patientnumber), np.zeros(patientnumber)
        tns_pancreas, fns_pancreas = np.zeros(patientnumber), np.zeros(patientnumber)

        with torch.no_grad():
            # -------------------------------------------- liver ---------------------------------------------------
            for data, labels, ids in self.liver_test_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]

                gt = labels.detach().cpu().numpy()
                y_out = torch.nn.functional.softmax(outputs[0], dim=1)
                pred = y_out.detach().cpu().numpy()
                seg = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    seg_ = seg[b_id:b_id + 1]
                    gt_ = gt[b_id:b_id + 1]

                    patientid = int(id_[6:10])  # liver_0001_1
                    if flag_liver[patientid] != 1:
                        flag_liver[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps_liver[patientid] += tp
                    fps_liver[patientid] += fp
                    tns_liver[patientid] += tn
                    fns_liver[patientid] += fn

                # torch.cuda.empty_cache()
                # break

            #  ---------------------------- calculate total liver loss and dice ----------------------------

            patients_liver = np.sum(flag_liver)
            tps_liver = tps_liver[flag_liver > 0]
            fps_liver = fps_liver[flag_liver > 0]
            tns_liver = tns_liver[flag_liver > 0]
            fns_liver = fns_liver[flag_liver > 0]
            dice_liver = 2 * tps_liver / (2 * tps_liver + fps_liver + fns_liver + 1e-40)  # p c
            mdice_liver = np.mean(dice_liver, axis=0)  # c

            # -------------------------------------------- kidney ---------------------------------------------------
            for data, labels, ids in self.kidney_test_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]

                gt = labels.detach().cpu().numpy()
                y_out = torch.nn.functional.softmax(outputs[1], dim=1)
                pred = y_out.detach().cpu().numpy()
                seg = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    seg_ = seg[b_id:b_id + 1]
                    gt_ = gt[b_id:b_id + 1]

                    patientid = int(id_[7:11])  # kidney_0001_1
                    if flag_kidney[patientid] != 1:
                        flag_kidney[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps_kidney[patientid] += tp
                    fps_kidney[patientid] += fp
                    tns_kidney[patientid] += tn
                    fns_kidney[patientid] += fn

                # torch.cuda.empty_cache()
                # break

            #  ---------------------------- calculate total kidney loss and dice ----------------------------
            patients_kidney = np.sum(flag_kidney)
            tps_kidney = tps_kidney[flag_kidney > 0]
            fps_kidney = fps_kidney[flag_kidney > 0]
            tns_kidney = tns_kidney[flag_kidney > 0]
            fns_kidney = fns_kidney[flag_kidney > 0]
            dice_kidney = 2 * tps_kidney / (2 * tps_kidney + fps_kidney + fns_kidney + 1e-40)  # p c
            mdice_kidney = np.mean(dice_kidney, axis=0)  # c

            # -------------------------------------------- pancreas ---------------------------------------------------
            for data, labels, ids in self.pancreas_test_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]

                gt = labels.detach().cpu().numpy()
                y_out = torch.nn.functional.softmax(outputs[2], dim=1)
                pred = y_out.detach().cpu().numpy()
                seg = np.argmax(pred, axis=1)  # b s h w -> b h w
                b, h, w = seg.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    seg_ = seg[b_id:b_id + 1]
                    gt_ = gt[b_id:b_id + 1]

                    patientid = int(id_[9:13])  # pancreas_0001_1
                    if flag_pancreas[patientid] != 1:
                        flag_pancreas[patientid] = 1

                    tp, fp, tn, fn = metrics.get_matrix(seg_, gt_)
                    tps_pancreas[patientid] += tp
                    fps_pancreas[patientid] += fp
                    tns_pancreas[patientid] += tn
                    fns_pancreas[patientid] += fn

                # torch.cuda.empty_cache()
                # break

            #  ---------------------------- calculate total pancreas loss and dice ----------------------------
            patients_pancreas = np.sum(flag_pancreas)
            tps_pancreas = tps_pancreas[flag_pancreas > 0]
            fps_pancreas = fps_pancreas[flag_pancreas > 0]
            tns_pancreas = tns_pancreas[flag_pancreas > 0]
            fns_pancreas = fns_pancreas[flag_pancreas > 0]
            dice_pancreas = 2 * tps_pancreas / (2 * tps_pancreas + fps_pancreas + fns_pancreas + 1e-40)  # p c
            mdice_pancreas = np.mean(dice_pancreas, axis=0)  # c

        self.model.to("cpu")

        Fed_dice = (mdice_liver + mdice_kidney + mdice_pancreas) / 3

        return Fed_dice, mdice_liver, mdice_kidney, mdice_pancreas

    def test_global_model_on_BCV(self):
        self.model.eval()
        self.model.cuda()

        patientnumber = 50  # record the patients
        flag = np.zeros(patientnumber)  # record the patients
        tps, fps = np.zeros((patientnumber, 3)), np.zeros((patientnumber, 3))
        tns, fns = np.zeros((patientnumber, 3)), np.zeros((patientnumber, 3))

        with torch.no_grad():
            for data, labels, ids in self.BCV_test_loader:
                data, labels = data.float().cuda(), labels.long().cuda()
                outputs = self.model(data)
                qkvs1, attns1 = outputs[3], outputs[4]

                gts = labels.detach().cpu().numpy()
                gt1 = np.zeros_like(gts)
                gt1[gts == 1] = 1  # liver binary label
                gt2 = np.zeros_like(gts)
                gt2[gts == 2] = 1  # kidney binary label
                gt3 = np.zeros_like(gts)
                gt3[gts == 3] = 1  # pancreas binary label

                y_out1 = torch.nn.functional.softmax(outputs[0], dim=1)
                pred1 = y_out1.detach().cpu().numpy()
                seg1 = np.argmax(pred1, axis=1)  # b s h w -> b h w

                y_out2 = torch.nn.functional.softmax(outputs[1], dim=1)
                pred2 = y_out2.detach().cpu().numpy()
                seg2 = np.argmax(pred2, axis=1)  # b s h w -> b h w

                y_out3 = torch.nn.functional.softmax(outputs[2], dim=1)
                pred3 = y_out3.detach().cpu().numpy()
                seg3 = np.argmax(pred3, axis=1)  # b s h w -> b h w

                b, w, h = seg2.shape

                for b_id in range(b):
                    id_ = ids[b_id]
                    patientid = int(id_[4:8])
                    flag[patientid] = 1

                    gt1_tmp = gt1[b_id:b_id + 1]
                    seg1_tmp = seg1[b_id:b_id + 1]
                    tp, fp, tn, fn = metrics.get_matrix(gt1_tmp, seg1_tmp)
                    tps[patientid, 0] += tp
                    fps[patientid, 0] += fp
                    tns[patientid, 0] += tn
                    fns[patientid, 0] += fn

                    gt2_tmp = gt2[b_id:b_id + 1]
                    seg2_tmp = seg2[b_id:b_id + 1]
                    tp, fp, tn, fn = metrics.get_matrix(gt2_tmp, seg2_tmp)
                    tps[patientid, 1] += tp
                    fps[patientid, 1] += fp
                    tns[patientid, 1] += tn
                    fns[patientid, 1] += fn

                    gt3_tmp = gt3[b_id:b_id + 1]
                    seg3_tmp = seg3[b_id:b_id + 1]
                    tp, fp, tn, fn = metrics.get_matrix(gt3_tmp, seg3_tmp)
                    tps[patientid, 2] += tp
                    fps[patientid, 2] += fp
                    tns[patientid, 2] += tn
                    fns[patientid, 2] += fn

                # break


        #  ---------------------------- calculate total dice ----------------------------
        patients = np.sum(flag)
        tps = tps[flag > 0, :]
        fps = fps[flag > 0, :]
        tns = tns[flag > 0, :]
        fns = fns[flag > 0, :]
        dice0 = 2 * tps / (2 * tps + fps + fns + 1e-40)  # p c
        mdices = np.mean(dice0, axis=0)  # c
        BCV_dice = np.mean(mdices, axis=0)  # 去掉背景类

        return BCV_dice, mdices[0], mdices[1], mdices[2]

    def fit(self):
        """Execute the whole process of the federated learning."""
        # self.val_results = {"val_loss": [], "dice": [], "dice_liver": [], "dice_kidney": [], "dice_pancreas": []}
        best_dice = 0.0
        best_BCV_dice = 0.0
        for r in range(self.num_rounds):
            self._round = r + 1

            self.train_federated_model()
            # break

            val_loss, dice, mdice_liver, mdice_kidney, mdice_pancreas = self.evaluate_global_model()

            self.writer.add_scalar('val_loss', val_loss, self._round)
            self.writer.add_scalar('val_average_dices', dice, self._round)
            self.writer.add_scalar('val_liver_dices', mdice_liver, self._round)
            self.writer.add_scalar('val_kidney_dices', mdice_kidney, self._round)
            self.writer.add_scalar('val_pancreas_dices', mdice_pancreas, self._round)



            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                                                        \n\t[Server] ...finished evaluation!\
                                                        \n\t=> Global val Loss: {val_loss:.4f}\
                                                        \n\t=> Global val Dice: {dice:.4f}, dice_liver: {mdice_liver:.4f}, dice_kidney: {mdice_kidney:.4f}, dice_pancreas: {mdice_pancreas:.4f}\n"
            print(message); logging.info(message)
            del message; gc.collect()


            if dice > best_dice:
                best_dice = dice

                save_path = os.path.join(self.checkpoints_path, str(best_dice))
                torch.save(self.model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

                Fed_dice, Fed_liver, Fed_kidney, Fed_pancreas = self.test_global_model_on_Fed()
                BCV_dice, BCV_liver, BCV_kidney, BCV_pancreas = self.test_global_model_on_BCV()

                message = f"[Round: {str(self._round).zfill(4)}] Test global model's performance...!\
                                                            \n\t[Server] ...finished testing!\
                                                            \n\t=> Fed_dice: {Fed_dice:.4f}, Fed_liver: {Fed_liver:.4f}, Fed_kidney: {Fed_kidney:.4f}, Fed_pancreas: {Fed_pancreas:.4f}\
                                                            \n\t=> BCV_dice: {BCV_dice:.4f}, BCV_liver: {BCV_liver:.4f}, BCV_kidney: {BCV_kidney:.4f}, BCV_pancreas: {BCV_pancreas:.4f}\n"
                print(message); logging.info(message)
                del message; gc.collect()


            if r % 5 == 0:
                state = {
                            'r': self._round,  # Saves the current number of iterations
                            'state_dict': self.model.state_dict(),  # save model parameters
                }
                save_path = os.path.join(self.checkpoints_tmp_path, str(r))
                torch.save(state, save_path + ".pth")



        # self.transmit_model()
