import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np
from os.path import join
import torch.nn.functional as F
from tqdm import tqdm
import collections

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)


        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask  # removed self-contrast


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # all prob (no self)


        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # log ( exp(positive)) - log (all)


        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mean_log_prob_pos = (ratio_mask * log_prob).sum(1) / mask.sum(1)


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()


        return loss

def generated_mask(labels, batch_size):

    index_sets = [torch.nonzero(labels[i] == labels, as_tuple=False) for i in range(0, 2* batch_size)]
    num_index = [item.shape[0] for item in index_sets]

    reals = [[item] * num_index[item] for item in range(0, len(num_index))]
    reals = [item for sublist in reals for item in sublist]

    num_values = list(torch.cat(index_sets).cpu().numpy())
    num_values = [arr.tolist()[0] for arr in num_values]

    idx = [i for i in range(0, len(reals)) if reals[i] == num_values[i]]
    reals = [i for j, i in enumerate(reals) if j not in idx]
    num_values = [i for j, i in enumerate(num_values) if j not in idx]

    return [reals, num_values]



class SimCLR_loss(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, temperature, batch_size):
        super(SimCLR_loss, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, out, out_1, out_2):


        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        #  simCLR loss (only first term of cvpr19 paper)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()



        return loss


def find_class_index(x):

    return [[x[j] == i for j in range(0, len(x))] for i in np.unique(x)]

class SimCLR_clear_loss(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, temperature, batch_size):
        super(SimCLR_clear_loss, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, out, out_1, out_2, labels):

        masked_labels = generated_mask(labels, self.batch_size)

        #  get mask
        sim_matrix_ori = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

        mask = (torch.zeros_like(sim_matrix_ori)).bool()
        mask[masked_labels[0], masked_labels[1]] = True


        # calculate sum of negs
        sim_matrix = sim_matrix_ori.masked_select(mask)
        class_index = find_class_index(masked_labels[0])
        sim_matrix = torch.stack([torch.sum(sim_matrix[item]) for item in class_index])


        # compute positive
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        #  compute loss
        loss = (- torch.log(pos_sim / sim_matrix)).mean()

        return loss


class SimCLR_clearspread_loss(nn.Module):
    ''' Compute the loss within each batch
    '''

    def __init__(self, temperature, batch_size):
        super(SimCLR_clearspread_loss, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, out, out_1, out_2, labels):

        masked_labels = generated_mask(labels, self.batch_size)

        #  get mask
        sim_matrix_ori = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

        mask = (torch.zeros_like(sim_matrix_ori)).bool()
        mask[masked_labels[0], masked_labels[1]] = True

        # neg mask: donot include only one sample case
        neg_mask = mask.clone()
        mask_index = torch.nonzero(torch.sum(neg_mask, dim=0)==1, as_tuple=False).tolist()
        mask_index = [item[0] for item in mask_index]
        neg_mask[mask_index,:] = torch.zeros((1, neg_mask.shape[1])).bool().cuda()

        # calculate sum of negs
        sim_matrix_masked = sim_matrix_ori.masked_select(neg_mask)
        neg_index = [i for i in masked_labels[0] if masked_labels[0].count(i)>1]
        class_index = find_class_index(neg_index)
        sim_matrix = torch.stack([torch.sum(sim_matrix_masked[item]) for item in class_index])

        # compute positive
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        pos_sim = pos_sim[np.unique(neg_index)]

        #  compute loss
        loss = (- torch.log(pos_sim / sim_matrix)).mean()

        return loss



class SimCLR_clearspread_loss_v3(nn.Module):
    ''' Compute the loss   domain-based nce + whole based spread
    '''

    def __init__(self, temperature, batch_size):
        super(SimCLR_clearspread_loss_v3, self).__init__()

        self.temperature = temperature
        self.batch_size = batch_size

    def forward(self, out, out_1, out_2, labels):

        masked_labels = generated_mask(labels, self.batch_size)
        # print (masked_labels)

        #  get mask
        sim_matrix_ori = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

        mask = (torch.zeros_like(sim_matrix_ori)).bool()
        mask[masked_labels[0], masked_labels[1]] = True

        # print(mask)
        # neg_all_mask = ~mask
        # print (neg_all_mask)
        # exit(0)
        # calculate sum of negs
        sim_matrix = sim_matrix_ori.masked_select(mask)
        class_index = find_class_index(masked_labels[0])
        sim_matrix = torch.stack([torch.sum(sim_matrix[item]) for item in class_index])

        # compute positive
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        # pos_sim = pos_sim[np.unique(neg_index)]

        #  compute loss
        # loss = (- torch.log(pos_sim / sim_matrix)).mean()
        #
        # # # ############## second term written by xmli
        #  all should be negative
        allneg_mask = (torch.ones_like(sim_matrix_ori).cuda() - torch.eye(2 * self.batch_size).cuda()).bool()
        allneg_sim_matrix = sim_matrix_ori.masked_select(allneg_mask).view(2 * self.batch_size, -1)
        # print ("aa", allneg_sim_matrix)

        all_div = allneg_sim_matrix.sum(1)
        Pon_div = all_div.repeat(self.batch_size * 2 - 1, 1)
        neg_sim = torch.div(allneg_sim_matrix, Pon_div.t())

        neg_sim = -neg_sim.add(-1)


        neg_index = [i for i in masked_labels[0] if masked_labels[0].count(i)>1]

        # posi, apply mask and sum
        pos = pos_sim / sim_matrix
        pos = pos[np.unique(neg_index)]
        neg_sim = neg_sim[np.unique(neg_index),:]


        neg_sim = torch.log(neg_sim).sum(dim=-1)
        #
        # # total loss: pos + negative
        loss = (- torch.log(pos) - (neg_sim)).mean()


        return loss

def kNN_linear(K, model, testloader, datasets_train, args, C):

    sigma = 0.07
    model.eval()
    top1 = 0.
    total = 0

    trainloader = torch.utils.data.DataLoader(datasets_train, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

    trainFeatures = torch.zeros((7, len(trainloader.dataset))).cuda()
    trainLabels = torch.zeros(len(trainloader.dataset)).cuda()
    for batch_idx, ((data, class_l, name), _) in enumerate(trainloader):

        data, class_l = data.cuda(), class_l.cuda()
        batchSize = data.shape[0]
        features, _ , _ = model(data)
        trainFeatures[:, batch_idx * batchSize : batch_idx * batchSize + batchSize] = features.data.t()
        trainLabels[batch_idx * batchSize : batch_idx * batchSize + batchSize] = class_l

    testfeature = []
    testlabel = []
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).long().cuda()
        for batch_idx, ((data, class_l, name), _) in enumerate(testloader):

            data, class_l = data.cuda(), class_l.cuda()
            batchSize = data.shape[0]
            features, _  , _ = model(data)

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)

            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi).long()

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(class_l.data.view(-1, 1))

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()

            total += class_l.size(0)

            testfeature.append(features.cpu().numpy())
            testlabel.append(class_l.cpu().numpy())

    return top1/float(total)



def kNN(K, model, testloader, datasets_train, args, C):

    sigma = 0.07
    model.eval()
    top1 = 0.
    total = 0

    #  load train dataset with test transformers
    # from datasets.dg_dataset import JigsawDataset, BaselineDataset, get_split_dataset_info, ConcatDataset, _dataset_info
    # datasets_train = []
    # for dname in args.source:
    #     name_train, name_val, labels_train, labels_val = get_split_dataset_info(
    #         join(args.data, 'data', 'txt_lists', '%s_train.txt' % dname), 0.1)
    #     train_dataset = BaselineDataset(name_train, labels_train, img_transformer=get_val_transformer(),
    #                                     two_transform=False)
    #     datasets_train.append(train_dataset)
    # datasets_train = ConcatDataset(datasets_train)
    trainloader = torch.utils.data.DataLoader(datasets_train, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False, drop_last=False)

    trainFeatures = torch.zeros((128, len(trainloader.dataset))).cuda()
    trainLabels = torch.zeros(len(trainloader.dataset)).cuda()
    for batch_idx, ((data, class_l, name), _) in enumerate(tqdm(trainloader)):

        data, class_l = data.cuda(), class_l.cuda()
        batchSize = data.shape[0]
        _, features, _ = model(data)
        trainFeatures[:, batch_idx * batchSize : batch_idx * batchSize + batchSize] = features.data.t()
        trainLabels[batch_idx * batchSize : batch_idx * batchSize + batchSize] = class_l

    testfeature = []
    testlabel = []
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).long().cuda()
        for batch_idx, ((data, class_l, name), _) in enumerate(tqdm(testloader)):

            data, class_l = data.cuda(), class_l.cuda()
            batchSize = data.shape[0]
            _, features, _ = model(data)

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)

            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi).long()

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(class_l.data.view(-1, 1))

            # find bad case
            # p = predictions.narrow(1, 0, 1)[0]
            # c = correct.narrow(1, 0, 1)[0]
            # for i in range(len(c)):
            #     if c[i] == False:
            #         print(name[i] +'wrong number:'+ str(p[i]))

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()

            total += class_l.size(0)

            testfeature.append(features.cpu().numpy())
            testlabel.append(class_l.cpu().numpy())


    return top1/float(total)




def multi_kNN(K, model, model_1, model_2, model_3, testloader, datasets_train, args, C):

    sigma = 0.07
    model_1.eval()
    model_2.eval()
    model_3.eval()
    top1 = 0.
    total = 0

    trainloader = torch.utils.data.DataLoader(datasets_train, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

    trainFeatures_1 = torch.zeros((128, len(trainloader.dataset))).cuda()
    trainFeatures_2 = torch.zeros((128, len(trainloader.dataset))).cuda()
    trainFeatures_3 = torch.zeros((128, len(trainloader.dataset))).cuda()
    trainLabels = torch.zeros(len(trainloader.dataset)).cuda()
    for batch_idx, ((data, class_l, name), _) in enumerate(tqdm(trainloader)):
        data, class_l = data.cuda(), class_l.cuda()
        batchSize = data.shape[0]

        _, feature_1, _ = model_1(data)
        _, feature_2, _ = model_2(data)
        _, feature_3, _ = model_3(data)
        trainFeatures_1[:, batch_idx * batchSize : batch_idx * batchSize + batchSize] = feature_1.data.t()
        trainFeatures_2[:, batch_idx * batchSize: batch_idx * batchSize + batchSize] = feature_2.data.t()
        trainFeatures_3[:, batch_idx * batchSize: batch_idx * batchSize + batchSize] = feature_3.data.t()
        trainLabels[batch_idx * batchSize : batch_idx * batchSize + batchSize] = class_l

    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).long().cuda()

        for batch_idx, ((data, class_l, name), _) in enumerate(tqdm(testloader)):
            data, class_l = data.cuda(), class_l.cuda()
            batchSize = data.shape[0]

            _, feature_1, _ = model_1(data)
            _, feature_2, _ = model_2(data)
            _, feature_3, _ = model_3(data)

            probs_1 = cal_accuracy(feature_1, trainFeatures_1, K, C, trainLabels, batchSize, retrieval_one_hot, sigma)

            probs_2 = cal_accuracy(feature_2, trainFeatures_2, K, C, trainLabels, batchSize, retrieval_one_hot,
                                         sigma)
            probs_3 = cal_accuracy(feature_3, trainFeatures_3, K, C, trainLabels, batchSize, retrieval_one_hot,
                                         sigma)

            probs = (probs_1+probs_2+probs_3)/3.0

            _, predictions = probs.sort(1, True)
            # print (predictions)
            #
            # print ("class", class_l.data.view(-1,1))

            # Find which predictions match the target
            correct = predictions.eq(class_l.data.view(-1, 1))


            top1 = top1 + correct.narrow(1, 0, 1).sum().item()

            total += class_l.size(0)

    return top1/float(total)


def cal_accuracy(feature, trainFeatures, K, C, trainLabels, batchSize, retrieval_one_hot, sigma):
    dist = torch.mm(feature, trainFeatures)
    print (dist.shape)
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    print ("yd", yd)
    print ("yi", yi)
    np.savetxt("indx.txt", yi.cpu().numpy(), delimiter=' ', fmt="%d")
    print(yd.shape, yi.shape)
    exit(0)

    candidates = trainLabels.view(1, -1).expand(batchSize, -1)
    retrieval = torch.gather(candidates, 1, yi).long()

    print ("candidates", candidates)
    print ("retri", retrieval)
    exit(0)
    retrieval_one_hot.resize_(batchSize * K, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
    yd_transform = yd.clone().div_(sigma).exp_()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)




    return probs


class SupMultipleConLoss(nn.Module):
    """Supervised Contrastive Learning from multiple domains: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupMultipleConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, queue0=None, queue1=None, queue2=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().cuda()


        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_count = features.shape[1]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T),
            self.temperature)
        # print("anchor_dot_contrast", anchor_dot_contrast)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        # print("logits_max", logits_max)
        logits = anchor_dot_contrast - logits_max.detach()
        # print ("logits", logits)

        # tile mask
        mask = mask.repeat(anchor_count, anchor_count)
        # print ("mask", mask)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # print ("logits_mask", mask)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # print ("self.tem", mean_log_prob_pos)
        print (self.temperature / self.base_temperature)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print ("lossss", loss)

        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss



# import torch.nn.functional as F
# input = np.random.rand(4, 2, 3)
# input = torch.from_numpy(input).cuda()
# input = F.normalize(input, dim=-1)
# input2 = np.random.rand(8, 2, 5)
# input2 = torch.from_numpy(input2).cuda()
# input2 = F.normalize(input2, dim=-1)
# queue = np.random.rand(5, 10)
# queue = torch.from_numpy(queue).cuda()
# queue = F.normalize(queue, dim=-1)
# queue2 = np.random.rand(5, 10)
# queue2 = torch.from_numpy(queue2).cuda()
# queue2 = F.normalize(queue2, dim=-1)
# queue3 = np.random.rand(5, 10)
# queue3 = torch.from_numpy(queue3).cuda()
# queue3 = F.normalize(queue3, dim=-1)
#
# labels = np.array([0, 2, 4, 4, 0, 2, 4, 4], dtype='float64')
# labels = torch.from_numpy(labels).cuda()
#
# print ("input anchor tensor", input2.shape)
# print ("labels", labels)
#
# input_resized = torch.reshape(input2[[0, 2, 3],:,:], (1,-1,5))
# queue = torch.reshape(queue.expand(1, 5, 10), (1, -1, 5))
# input_resized = torch.cat([input_resized, queue], dim=1)
#
# input_resized2 = torch.reshape(input2[[1, 4, 5],:,:], (1,-1,5))
# queue2 = torch.reshape(queue2.expand(1, 5, 10), (1, -1, 5))
# input_resized2 = torch.cat([input_resized2, queue2], dim=1)
#
# input_resize = torch.cat([input_resized, input_resized2])
# label_resize = np.array([0, 1], dtype='float64')
# label_resize = torch.from_numpy(label_resize).cuda()
#
# # sup_criterion = SupConLoss()
# supmultiple_criterion = SupConLoss()
# # # criterion1 = SimCLR_loss(0.1, 4)chu
#
# # loss = criterion1(torch.cat([input, input2], 0), input, input2)
# # print (input.shape, input2.shape)
# # loss = criterion2(torch.cat([input, input2], 0), input, input2, labels)
# # print (loss)
# # loss = supmultiple_criterion(input2, labels, queue, queue2, queue3)
#
# loss = supmultiple_criterion(input2, labels)
