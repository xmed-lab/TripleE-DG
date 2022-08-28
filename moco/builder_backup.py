# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False,args=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(classes=10,args=args)
        self.encoder_k = base_encoder(classes=10, args=args)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_didx", torch.randn(K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_didx", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, didx):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        ptr_didx = int(self.queue_ptr_didx)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_didx[ptr_didx:ptr_didx + batch_size] = didx

        ptr = (ptr + batch_size) % self.K  # move pointer
        ptr_didx = (ptr_didx + batch_size) % self.K  # move pointer
        # print ("batch_size", batch_size, keys.T.shape, ptr)

        self.queue_ptr[0] = ptr
        self.queue_ptr_didx[0] = ptr_didx



    def forward(self, im_q, im_k, domain_idx, trainflag):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        _, [_, q], _ = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)

        _, [inv_q, _], _ = self.encoder_q(torch.cat([im_q, im_k],0))

        if trainflag:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                _, [inv_k, k], _ = self.encoder_k(im_k)  # keys: NxC
                # k = nn.functional.normalize(k, dim=1)

            # # ~~~~~~~~~~~~~~~~ ori ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # compute logits
            # # Einstein sum is more intuitive
            # # positive logits: Nx1
            # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # # negative logits: NxK
            # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            #
            # # logits: Nx(1+K)
            # logits = torch.cat([l_pos, l_neg], dim=1)
            # # apply temperature
            # logits /= self.T
            # # labels: positive key indicators
            # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            # # dequeue and enqueue
            # self._dequeue_and_enqueue(k)

            # return inv_q, q, logits, labels

            # ~~~~~~~~~~~~~~~~ multi-positive ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # concate input
            p = torch.reshape(domain_idx, (2, 4, 4))
            c = p.permute((1, 0, 2)).reshape(4, -1).T

            a = self.queue_didx.clone().detach()
            b = torch.reshape(a, (int(self.K/16), 4, 4))  # how many block pushes, each block repeat times, four sampled ID
            labels = b.permute((1,0,2)).reshape(4,-1).T
            newlabels = torch.cat([c[:,0], labels[:,0]],0)  # 72: 8+64

            # new features: 128*256 -> N*4*128
            temp = self.queue.clone().detach()
            b = torch.reshape(temp, (-1, 128, 4))
            temp = b.permute((0,2,1))

            qk = torch.cat([q,k],0)

            b = torch.reshape(qk, (-1,128,4))
            temp2 = b.permute((0,2,1))
            newfeatures = torch.cat([temp2, temp],0)  # 72,4,128



            loss = multi_loss(newfeatures, newlabels)
            # enquee
            self._dequeue_and_enqueue(k, domain_idx[:16])

            return inv_q, q, loss, _


        else:
            return inv_q


def multi_loss(features, labels):

    batch_size = features.shape[0]

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().cuda()
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        0.07)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    
    return loss

# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
# 
#     output = torch.cat(tensors_gather, dim=0)
#     return output