# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, args=False):
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
        self.encoder_q = base_encoder(classes=7, args=args)
        self.encoder_k = base_encoder(classes=7, args=args)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        for num in range(0, 7):
            self.register_buffer("queue_"+str(num), torch.randn(dim, K))
            self.register_buffer("queue_ptr_"+str(num), torch.zeros(1, dtype=torch.long))

        self.queue_list = [self.queue_0, self.queue_1, self.queue_2, self.queue_3, self.queue_4, self.queue_5, self.queue_6]
        self.queue_list_ptr = [self.queue_ptr_0, self.queue_ptr_1, self.queue_ptr_2, self.queue_ptr_3, self.queue_ptr_4, self.queue_ptr_5, self.queue_ptr_6]


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)


        # batch_size = keys.shape[0]

        # ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print (targets)
        for i in range(0, keys.shape[0]):
            nt = targets[i].cpu().numpy()
            ptr = int(self.queue_list_ptr[nt])
            self.queue_list[nt][:, ptr:ptr + 1] = keys[i:i+1,:].transpose(1,0)
            ptr = (ptr + 1) % self.K  # move pointer
            self.queue_list_ptr[nt][0] = ptr

        # print (self.queue_list_ptr)


    def forward(self, im_q, im_k, targets):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        logits_q, q, _ = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            logits_k, k, _ = self.encoder_k(im_k)  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        assert im_k.shape == im_q.shape
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = []
        for t in range(0, targets.shape[0]):
            nt = targets[t].cpu().numpy()
            l_neg.append(torch.einsum('c,ck->k', [q[t,:], self.queue_list[nt].cuda().clone().detach()]))
        l_neg = torch.stack(l_neg)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, targets)

        return logits, labels, logits_q


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
