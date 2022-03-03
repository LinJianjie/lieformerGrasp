import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import components.transformerLinearNet.t_constant
from components.netUtils import NetUtil, CPointNet, CPointNetLinear
from components.dgcnn.dgcnn_utils import DGCNN
from enum import Enum


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    return torch.tril(torch.ones(1, size, size))


def generate_local_square_map_mask(chunk_size, chunk_size_2=None, attention_size=None, mask_future=False):
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    if chunk_size_2 is not None:
        local_map = np.empty((chunk_size, chunk_size_2))
    else:
        local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map)


class transformer_last_layer(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(transformer_last_layer, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1) # remove the the softmax
        return self.proj(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, use_cmlp=False):
        super(Embeddings, self).__init__()
        self.pn = CPointNetLinear([3, 128, 128, 256, 512])

        # self.linear = nn.Linear(512, d_model)

    def forward(self, x):
        # return self.lut(x) * math.sqrt(self.d_model)
        # instead using embeding, we use linear transformation maps the value to a high dimensional
        B, N, D = x.shape
        x = x.permute(0, 2, 1)
        x = self.pn(x)
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(B, 1, -1)
        # return self.linear(x)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def l2_loss(input_x, target, size_average=True):
    """ L2 Loss without reduce flag.
    Args:
        input_x (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input_x - target), 2))
    else:
        return torch.mean(torch.mean(torch.pow((input_x - target), 2), dim=-1), dim=-1)


def l1_loss(input, target):
    """ L1 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


def consistency_cos_similarity(input_x, target_x, size_average=True):
    norm_input_x = torch.norm(input_x, dim=-1)
    norm_target_x = torch.norm(target_x, dim=-1)
    input_target = torch.sum(input_x * target_x, dim=-1)
    valid_label1 = Variable(torch.Tensor(target_x.shape[0], target_x.shape[1]).fill_(1.0), requires_grad=False).cuda()
    z = torch.pow(input_target / (norm_input_x * norm_target_x) - valid_label1, 2)
    if size_average:
        return torch.mean(z)
    else:
        return torch.mean(z, dim=-1)


def add_noise_data(data):
    b, D, C = data.shape
    x = copy.deepcopy(data)
    select_data = x[:, 1:-1, :]
    noise = Variable(
        torch.Tensor(np.random.normal(0, 1, (select_data.shape[0], select_data.shape[1], select_data.shape[2]))),
        requires_grad=False).cuda()
    select_data_with_noise = select_data + noise
    final_data = torch.cat([data[:, 0, :].view(b, 1, C), select_data_with_noise, data[:, -1, :].view(b, 1, C)], dim=1)
    return final_data


def normalized_data(data):
    b, D, C = data.shape
    select_data = data[:, 1:-1, :]
    if use_shift_data:
        # u = torch.mean(select_data, dim=1, keepdim=True)
        # normalized_select_data = (select_data)
        # # normalized_select_data = select_data
        # max_u = torch.max(torch.abs(normalized_select_data), dim=1, keepdim=True)[0]
        # index = max_u == torch.zeros_like(max_u)
        # max_u[index] = 1
        # if inside_zero_one:
        #     min_u = torch.min(normalized_select_data, dim=1, keepdim=True)[0]
        #     dividi_item = max_u - min_u
        #     index = dividi_item == torch.zeros_like(dividi_item)
        #     dividi_item[index] = 1
        #     normalized_select_data = (normalized_select_data - min_u) / (dividi_item)
        # else:
        #     normalized_select_data = normalized_select_data / max_u
        if select_gripper and select_robot:
            return data
        if select_gripper and not select_robot:
            return data[:, :, :2]
        if not select_gripper and select_robot:
            return data[:, :, 2:]
    else:
        u = torch.mean(select_data, dim=1, keepdim=True)
        normalized_select_data = (select_data - u)
        # normalized_select_data = select_data
        final_data = torch.cat([data[:, 0, :].view(b, 1, C), normalized_select_data, data[:, -1, :].view(b, 1, C)],
                               dim=1)
        return final_data

    # print(data[:, 0, :].view(b, 1, C).shape)
    # print(data[:, -1, :].view(b, 1, C).shape)


def add_start_tok(data):
    start_character = torch.ones([data.shape[0], 1, data.shape[2]]) * sos
    start_character = start_character.view(data.shape[0], 1, data.shape[2]).to(data.device)
    final_data = torch.cat([start_character, data], dim=1)
    return final_data


def get_target_input(tgt_input):
    return tgt_input[:, :-1]


def get_target_gt(tgt_input):
    return tgt_input[:, 1:]


def save_checkpoint(state, best_model_name):
    save_path = os.path.join(os.path.dirname(__file__), "../../checkpoints", best_model_name)
    torch.save(state, save_path)


def predictive(loss_g_all, max_threshold, min_threshold, mean_threshold):
    max_threshold_all = torch.ones_like(loss_g_all).to(loss_g_all.device) * max_threshold
    min_threshold_all = torch.ones_like(loss_g_all).to(loss_g_all.device) * min_threshold
    mean_threshold_all = torch.ones_like(loss_g_all).to(loss_g_all.device) * mean_threshold
    if use_max_threshold:
        anomalyRES = loss_g_all < max_threshold_all
    else:
        anomalyRES = loss_g_all < mean_threshold_all
    # anomalyRES = torch.ones_like(loss_g_all).to(loss_g_all.device)
    # for i in range(loss_g_all.shape[0]):
    #     if loss_g_all[i] < min_threshold:
    #         anomalyRES[i] = 1
    #     elif loss_g_all[i] > max_threshold:
    #         anomalyRES[i] = 0
    #     else:
    #         pro = torch.rand(1).item()
    #         if pro > 0.5:
    #             anomalyRES[i] = 1
    #         else:
    #             anomalyRES[i] = 0
    return anomalyRES


def vote_accuracy(true_list, pred_list, feature_label):
    index_normal = (true_list == 1)
    index_anomaly = (true_list == 0)
    # for normal test
    normal_feature_label = feature_label[index_normal]
    pred_normal_list = pred_list[index_normal]

    unique_normal_label = np.unique(normal_feature_label)
    n_normal_label = unique_normal_label.shape[0]

    normal_vote_list = []
    for i in range(n_normal_label):
        check_index = normal_feature_label == unique_normal_label[i]
        select_res = pred_normal_list[np.squeeze(check_index)]
        prob = np.sum(1 - select_res) / select_res.shape[0]
        if prob > 0.5:
            normal_vote_list.append(0)
        else:
            normal_vote_list.append(1)
    normal_vote_list = np.asarray(normal_vote_list)
    normal_accuracy = 1 - np.sum(1 - normal_vote_list) / n_normal_label
    # for anomaly test
    anomaly_feature_label = feature_label[index_anomaly]
    pred_anomaly_list = pred_list[index_anomaly]

    unique_anomaly_label = np.unique(anomaly_feature_label)
    n_anomaly_label = unique_anomaly_label.shape[0]

    anomaly_vote_list = []

    for i in range(n_anomaly_label):

        check_index = anomaly_feature_label == unique_anomaly_label[i]
        select_res = pred_anomaly_list[np.squeeze(check_index)]
        prob = np.sum(select_res) / select_res.shape[0]
        if prob > 0.5:
            anomaly_vote_list.append(1)
        else:
            anomaly_vote_list.append(0)
    anomaly_vote_list = np.asarray(anomaly_vote_list)
    anomaly_accuracy = 1 - np.sum(anomaly_vote_list) / n_anomaly_label
    # whole_accuracy = (normal_accuracy * normal_feature_label.shape[0] + anomaly_accuracy * anomaly_feature_label.shape[
    #     0]) / (normal_feature_label.shape[0] + anomaly_feature_label.shape[0])
    whole_accuracy = (normal_accuracy + anomaly_accuracy) / 2.0
    return whole_accuracy, normal_accuracy, anomaly_accuracy


class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        print(similarity_matrix)
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        print("l_pos: ", l_pos)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        print("r_pos: ", r_pos)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        print("positive: \n", positives)
        print("similarity_matrix:\n ", similarity_matrix)
        print("self.mask_samples_from_same_repr.shape: \n", self.mask_samples_from_same_repr)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        print("negative: \n", negatives)

        logits = torch.cat((positives, negatives), dim=1)
        print("logits: ", logits.shape)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        print("labels: ", labels.shape)
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0, use_eu_dist=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.use_eu_dist = use_eu_dist

    def forward(self, output1, output2, label):
        # Label 1 --> indicates the same data
        # Label 0 --> indicates the different data
        # Find the pairwise distance or eucledian distance of two output feature vectors
        if self.use_eu_dist:
            eu_dist = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean(
                (label) * torch.pow(eu_dist, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - eu_dist, min=0.0),
                                                                          2))

            return loss_contrastive
        else:
            similarity_dist = F.cosine_similarity(output1, output2)
            loss_contrastive = torch.mean(
                label * torch.pow(1 - similarity_dist, 2) + (1 - label) * torch.pow(similarity_dist, 2))
            return loss_contrastive


def makedirs_checkout():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "../../checkpoints")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../checkpoints"))


def makedirs_log():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "../../logs")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../logs"))


if __name__ == '__main__':
    # y = subsequent_mask(4)
    y = torch.rand(2, 5, 3)
    y1 = normalized_data(y)
    # print(y)
    loss = NTXentLoss(2, 0.7, use_cosine_similarity=True)
    # print(loss.mask_samples_from_same_repr)
    zj = torch.rand(2, 10).cuda()
    zi = torch.rand(2, 10).cuda()
    loss(zj, zi)
