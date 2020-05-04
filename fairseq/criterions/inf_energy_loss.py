# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

import torch

import random
random.seed(1)


def gumbel_softmax(logits, temperature=1.0, withnoise=True, hard=True, eps=1e-10):

             def sample_gumbel(shape, eps=1e-10):
                 U = torch.rand(shape).cuda()
                 return -torch.log(-torch.log(U + eps) + eps)


             if withnoise:
                   gumbels = sample_gumbel(logits.size())
                   y_soft = ((logits + gumbels)*1.0/temperature).softmax(2)
             else:
                   y_soft = (logits*1.0/temperature).softmax(2)

             index = y_soft.max(dim=-1, keepdim=True)[1]
             y_hard = torch.zeros_like(logits).scatter_(2, index, 1.0)
             ret = (y_hard - y_soft).detach() + y_soft
             if hard:
                  return ret
             else:
                  return y_soft


def STL(logits):

             index = logits.max(dim=-1, keepdim=True)[1]
             y_hard = torch.zeros_like(logits).scatter_(2, index, 1.0)
             ret = (y_hard - logits).detach() + logits

             return ret




@register_criterion('Inf_Energy_Loss')
class Inf_Energy_Loss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.feed_type = args.feed_type
        self.alpha = args.alpha
        self.teacher_forcing = args.teacher_forcing
       



    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        #parser.add_argument('--feed_type', default =2 ,type=int, metavar='N',
        #                    help='computing the energy')

        parser.add_argument('--alpha', default =0 ,type=float, metavar='D',
                            help='weigh for local loss')

        parser.add_argument('--teacher_forcing', default =0 ,type=float, metavar='D',
                            help='ratio for feeding golden tokens into the energy') 



    def forward(self, model, model_E, sample, train=True, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
       
        net_output = model(**sample['net_input'])
        energy, discrete_energy, nll_loss, length_loss, ntokens = self.compute_loss(model, model_E, net_output, sample, train=train, reduce=reduce)
        sample_size = ntokens #TODO why not merge ntokens and sample_size? what is the difference?
        logging_output = {
            'loss': utils.item(energy.data) if reduce else energy.data,
            'discrete_loss': utils.item(discrete_energy.data) if reduce else discrete_energy.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return energy, sample_size, logging_output

    def compute_loss(self, model, model_E, net_output, sample, train=True, reduce=True):

        lprobs0 = model.get_normalized_probs(net_output, log_probs=True)      
        #p_output = model.get_normalized_probs(net_output, log_probs=False)

        lprobs = lprobs0.view(-1, lprobs0.size(-1))


        target = model.get_targets(sample, net_output).view(-1, 1)

        non_pad_mask = sample['target'].ne(self.padding_idx)
        end_padding = sample['target'].eq(2)
        non_padding = non_pad_mask & (~end_padding)


        length_lprobs = net_output[1]['predicted_lengths']
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1) #TODO doesn't work for dynamic length. change to eos-based method.
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask.view(-1,1)]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask.view(-1,1)]
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = self.alpha*((1. - self.eps) * nll_loss + eps_i * smooth_loss) + length_loss



        inf_pred= torch.argmax(lprobs0, dim=-1)
        one_hot = torch.cuda.ByteTensor(inf_pred.size(0), inf_pred.size(1), lprobs0.size(-1)).zero_()
        one_hot = one_hot.scatter_(2, inf_pred.unsqueeze(2), 1)


        prev_in = sample['net_input']['prev_output_tokens']
        prev_in_log = torch.cuda.FloatTensor(lprobs0.size()).fill_(-1e10)
        #if train:
        #     prev_in_log = prev_in_log.scatter_(2, prev_in.unsqueeze(2), 0)
        #     pad_mask = prev_in.eq(4)
        #     lprobs0 = torch.where(pad_mask.unsqueeze(2).repeat(1, 1, lprobs0.size(2)), lprobs0, prev_in_log)


        if train==False:
             lprobs0 = prev_in_log.scatter_(2, inf_pred.unsqueeze(2), 0) 
                

        p_output = lprobs0*(non_padding.unsqueeze(2).repeat(1, 1, lprobs0.size(2)).float())



        newsample = {'src_tokens':sample['net_input']['src_tokens'], 'src_lengths': sample['net_input']['src_lengths'], 'prev_output_tokens':(p_output, sample['net_input']['prev_output_tokens'])} 
        final_output = model_E(**newsample)

        
        scores_s = model_E.get_normalized_probs(final_output, log_probs=True)
    

        if self.feed_type==0:
                  ## all the word distributions                                
                  xent =  -torch.sum(scores_s*torch.exp(p_output), dim = -1)
        elif self.feed_type==1:
                  #print('straight-through Gumbel-Softmax')
                  xent =  -torch.sum(scores_s*gumbel_softmax(lprobs0, temperature=1.0), dim = -1)
        elif self.feed_type==2:
                  # ST (probalities)
                  xent =  -torch.sum(scores_s*gumbel_softmax(lprobs0, temperature=1.0, withnoise=False), dim = -1)
        elif self.feed_type==3:
                  # ST (log probalities)
                  xent =  -torch.sum(scores_s*STL(lprobs0), dim = -1)
        elif self.feed_type==4:
                  # Gumbel-Softmax
                  xent =  -torch.sum(scores_s*gumbel_softmax(lprobs0, temperature=1.0, withnoise=True, hard=False), dim = -1)
          
        ce_train = scores_s.masked_select(one_hot.bool())
        #######exclude the last one
        #ce_train = -ce_train.masked_select(non_padding.view(-1)).sum()- torch.gather(scores_s[:,:,3], 0, end_index.view(1, -1)).sum()

        ####### include the energy at the last position
        disctre_energy = -ce_train.masked_select(non_padding.contiguous().view(-1)).sum() - scores_s[:,:,2].masked_select(end_padding).sum()

        
        energy = xent.masked_select(non_padding).sum() - scores_s[:,:,2].masked_select(end_padding).sum() + loss

        return energy, disctre_energy, nll_loss, length_loss, non_pad_mask.sum().data.item()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'discrete_loss': sum(log.get('discrete_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
