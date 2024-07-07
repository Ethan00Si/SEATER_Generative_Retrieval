import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import copy
import numpy as np
from typing import Optional

from .inputs import *


class EncoderDecoder(nn.Module):
    """
    Following a standard Encoder-Decoder architecture
    """

    def __init__(self, config):
        super().__init__()

        self.src_embed = None
        self.src_pos_embed = PositionalEmbedding(config['max_reco_his'], config['d_model'])

        self.encoder = Encoder(
            N=config['Encoder']['N_layer'], d_model=config['d_model'],
            nhead=config['Encoder']['nhead'], inner_dim=config['Encoder']['inner_dim'],
            dropout=config['Encoder']['dropout']
        )
        self.decoder = CausalTransformerDecoder(
            N=config['Decoder']['N_layer'], d_model=config['d_model'],
            nhead=config['Decoder']['nhead'], inner_dim=config['Decoder']['inner_dim'],
            dropout=config['Decoder']['dropout']
        )
        
        self.tgt_embed = None
        self.tgt_pos_embed = None

    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        "Take in and process masked src and target sequences."

        return self.decode(
            tgt, self.encode(src, src_pad_mask), src_pad_mask, tgt_mask
        )
    
    def encode(self, src, src_pad_mask):
        
        src = self.src_pos_embed( self.src_embed(src) )

        return self.encoder(
            src, src_pad_mask
        )
    

    def decode(self, 
                      tgt: torch.Tensor, 
                      memory: torch.Tensor, 
                      src_pad_mask: torch.Tensor, 
                      tgt_mask: torch.Tensor, 
                      cache: Optional[torch.Tensor] = None,
                      ):
        '''
        efficient causal decoding. Only used for inference
        '''    
        tgt = self.tgt_pos_embed( self.tgt_embed(tgt) )

        return self.decoder(
            tgt, memory, 
            memory_key_padding_mask = src_pad_mask,
            tgt_mask = tgt_mask, 
            cache = cache
        )

    
    def train_step(self, batch):
        raise NotImplementedError()
        
    def predict_step(self, batch, topK=50):
        raise NotImplementedError()
    
class Encoder(nn.Module):
    '''
    A standard transformer encoder
    '''

    def __init__(self, N, d_model, nhead, inner_dim, dropout, activation='gelu'):
        '''
        Args:
            N : number of layers
            d_model : dimension of model
            nhead: number of heads
            inner_dim: dim of feedforward network
            dropout: dropout rate
            activation: function
        '''
        super(Encoder, self).__init__()
        layer = nn.TransformerEncoderLayer(
            batch_first=True, d_model=d_model, nhead=nhead, dim_feedforward=inner_dim,
            dropout=dropout, activation=activation
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


    def forward(self, x, mask):
        '''
        Pass the input (and mask) through each layer in turn.
        Args:
            x (torch.tensor) of shape (B, S, D)
            mask (torch.tensor) of shape (B, S): a True value indicates that the corresponding key value 
                                                will be ignored for the purpose of attention.
        '''
        for layer in self.layers:
            x = layer(x, src_key_padding_mask = mask)
        return x

class CausalTransformerDecoder(nn.Module):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).

    Referring to https://github.com/alex-matton/causal-transformer-decoder/blob/master/causal_transformer_decoder/model.py
    """
    def __init__(self, N, d_model, nhead, inner_dim, dropout, activation='gelu'):
        '''
        Args:
            N : number of layers
            d_model : dimension of model
            nhead: number of heads
            inner_dim: dim of feedforward network
            dropout: dropout rate
            activation: function
        '''
        super().__init__()
        layer = CausalTransformerDecoderLayer(
            batch_first=True, d_model=d_model, nhead=nhead, dim_feedforward=inner_dim,
            dropout=dropout, activation=activation
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt (Tensor):  bsz x current_len_output x hidden_dim
            memory (Tensor): bsz x len_encoded_seq x hidden_dim
            cache (Optional[Tensor]):
                n_layers x bsz x (current_len_output - 1) x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): bsz x current_len_output x hidden_dim
            cache (Optional[Tensor]): n_layers x bsz x current_len_output x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        if self.training:
            if cache is not None:
                raise ValueError("cache parameter should be None in training mode")
            for i, mod in enumerate (self.layers):
                output = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )

            return output

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(
                        output, 
                        memory,
                        memory_key_padding_mask = memory_key_padding_mask
                    ) # bsz x current_length x hidden_dim
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=1)

        if cache is not None:
            # n_layers x bsz x current_len_output x hidden_dim
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=2)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation=F.relu, layer_norm_eps=0.00001, batch_first=False,
                 norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                         activation, layer_norm_eps, batch_first,
                         norm_first, device, dtype)
    # @torchsnooper.snoop()
    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer:  bsz x seq_len x hidden_dim
                If eval mode: embedding of last token: bsz x 1 x hidden_dim
        """

        if self.training:
            return super().forward(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.
        tgt_last_tok = tgt[:, -1:, :] #batch first

        # self attention part
        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        tgt_last_tok = self.norm2(tgt_last_tok + self._mha_block(tgt_last_tok, memory, memory_mask, memory_key_padding_mask))
        tgt_last_tok = self.norm3(tgt_last_tok + self._ff_block(tgt_last_tok))

        return tgt_last_tok

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, dim):
        '''
        Absolute position embeddings
        '''
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        pos_emb = self.pe.weight[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        return x + pos_emb


   
class Generator(EncoderDecoder):
    def __init__(self, config):
        '''implement generation using constrained beam search'''
        super().__init__(config)

        self.softmax = nn.Softmax(dim=-1)

        self.constrained_prefix_tree = None # to do: init
        self.log_softmax = nn.LogSoftmax(dim=-1)


    def _init_prefix_mask(self, constrained_tree, device):
        # not parameters
        '''
        constrained_tree:
            an array with shape (#all nodes, vocab_size)  
                * vocab_size: all allowed next tokens, padded to the same length
                * end token: As for end token, only end token is allowed to be the next.
                * if an allowed token is the pad token, it should be ignored in generation process.
        '''
        self.constrained_prefix_tree = torch.tensor(constrained_tree, dtype=torch.long, device=device)

    def predict_step(self, batch, topK):
        _, src = batch #uid not used
        src_pad_mask = torch.where(src[:,:,0]==0, 1, 0).bool() # B, S

        candidate, scores = self.generate_topk(src, src_pad_mask, self.beam_size)

        scores = torch.exp(scores) # log_prob -> prob

        candidate, scores = candidate[:, :topK, :], scores[:, :topK]

        pred_items_id = torch.sum(candidate != self.tgt_embed.end_token, dim=-1, keepdim=True)
        pred_items_id = torch.gather(candidate, dim=-1, index=pred_items_id-1).squeeze(-1)

        return scores, pred_items_id

    # @torchsnooper.snoop()
    def generate_topk(self, 
                    src: torch.tensor, 
                    src_pad_mask: torch.tensor,
                    topK: int, 
                    ):
        '''
        Beam search.
        Generate the next item with top K highest prob.

        Args:
            src of shape (B, S)
            src_pad_mask of shape (B, S)
            topK: beam size
        '''
        # [B, S, dim]
        memory = self.encode(src, src_pad_mask)
        # [B, beam, max_seq_len], [B, beam]
        candidate_results, scores = self.generate_start_token(
            beam_size=topK, bs=src.size(0), device=src.device
        )
        # [B*beam, S, dim]
        memory = memory.unsqueeze(1).repeat(1, topK, 1, 1).reshape(-1, memory.size(1), memory.size(-1))
        # [B*beam, S]
        src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, topK, 1).reshape(-1, memory.size(1))

        cache = None
        for level in range(1, self.max_len):
            tgt = candidate_results[:, :, :level] #[B, beam, len]
            tgt = tgt.reshape(-1, tgt.size(-1))
            decoder_hidden, cache = self.decode(
                tgt, memory, src_pad_mask, subsequent_mask(tgt.size(1)).to(src.device),
                cache = cache
            )
            candidate_results, scores, cache = self.generate_per_level(
                decoder_hidden[:, -1], candidate_results, scores, level, topK, cache
            )

        return candidate_results, scores 

    def generate_start_token(self, 
                             beam_size: int, 
                             bs: int,
                             device: torch.device
                             ):
        # [batch, beam_size, max_seq_len]
        candidate_results = torch.ones(
            (bs, beam_size, self.max_len), dtype=torch.long, device=device
        ) * (-100) # -100 denotes meaningless token
        candidate_results[:, :, 0] = self.tgt_embed.start_token
        beam_scores = torch.zeros(
            (bs, beam_size), device=device
        )

        return candidate_results, beam_scores
      
    def generate_per_level(self, 
                           decoder_hidden: torch.tensor, 
                           candidate_results: torch.tensor,
                           beam_scores: torch.tensor, 
                           level: int, 
                           beam_size: int,
                           cache: torch.tensor
                           ):
        '''
        Args:
            decoder_hidden of shape (batch*beam_size, dim)
            candidate_results of shape (batch, beam_size, max_length)
            beam_scores of shape (batch, beam_size)
            level: current level of the prefix tree
            beam_size: topk
        '''
        bs = candidate_results.size(0)

        # [batch, beam_size, vocab, dim], [batch, beam_size, vocab]
        next_candiates_emb, next_candiates_idx = self.select_next_level_candidate(candidate_results[:, :, level-1]) 
        # [batch, beam_size, vocab]   True means invalid 
        allowed_next_candidates_mask = next_candiates_idx == self.tgt_embed.pad_token

        # [batch, beam_size, vocab]
        prob_for_all_candidates = torch.einsum(
            'bsvd,bsd->bsv', next_candiates_emb, decoder_hidden.reshape(bs, beam_size, -1)
        )
        prob_for_all_candidates = self.log_softmax(
            prob_for_all_candidates.masked_fill_(allowed_next_candidates_mask, float('-inf'))
        ) # mask all the not-allowed positions

        vocab_size = self.tgt_embed.vocab_size

        # [batch, beam_size, vocab]
        scores = beam_scores.unsqueeze(-1) + prob_for_all_candidates
        # Get the best beam_size candidates from beam_size * vocab candidates.
        # [batch, beam_size]
        if level != 1:
            scores, best_k_idx = scores.view(bs, -1).topk(k = beam_size, dim = -1)
        else:
            tmp_scores, tmp_best_k_idx = scores[:,0,:].topk(k = min(beam_size, vocab_size), dim = -1)
            scores = torch.ones(bs, beam_size, device=scores.device) * float('-inf')
            best_k_idx = torch.zeros(bs, beam_size, device=scores.device, dtype=torch.long)
            if beam_size > vocab_size:
                best_k_idx[:, :vocab_size] = tmp_best_k_idx
                scores[:, :vocab_size] = tmp_scores
                best_k_idx[:, vocab_size:] = tmp_best_k_idx[0,0]
            else:
                scores = tmp_scores
                best_k_idx = tmp_best_k_idx

        # [batch, beam]
        best_k_r_idxs = torch.div(best_k_idx, vocab_size, rounding_mode='floor')
        # Get the corresponding positions of the best k candidiates.
        best_k_idx = torch.gather(next_candiates_idx.reshape(bs, -1), dim=-1, index=best_k_idx)

        # Copy the corresponding previous tokens.
        candidate_results[:, :, :level] = torch.gather(
            candidate_results, dim=1, index=best_k_r_idxs.unsqueeze(-1).repeat(1, 1, level)
        )
        # Set the best tokens in this beam search step
        candidate_results[:, :, level] = best_k_idx

        # reorder cache
        # [n_layers, bs*beam, level, dim] 
        for i in range(cache.shape[0]):
            tmp_cache = torch.gather(
                cache[i].reshape(bs, beam_size, level, -1), dim=1,
                index=best_k_r_idxs.reshape(bs, beam_size, 1, 1)\
                    .repeat(1, 1, level, cache[i].size(-1))
            )
            cache[i] = tmp_cache.reshape(bs*beam_size, level, -1)
       
        return candidate_results, scores, cache
    
    def select_next_level_candidate(self, cur_nodes:torch.tensor):
        '''
        Args:
            * cur_nodes (torch.tensor) : current nodes. we need to select candidates from their children nodes
        Returen:
            * candidate_emb (torch.tensor): candidate embeddings
            * candidate_idx (torch.tensor): candidate IDs
        '''
        candidate_idx = F.embedding(cur_nodes, self.constrained_prefix_tree) # [beam_size, vocab]
        candidate_emb = self.tgt_embed(candidate_idx) #[beam_size, vocab, dim]

        return candidate_emb, candidate_idx
    

class SEATER(Generator):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.tgt_embed = OURS_item_feat(config)
        self.tgt_pos_embed = PositionalEmbedding(config['decoder_index']['max_len'], config['d_model'])
        self.src_embed = self.tgt_embed

        self.max_len = config['decoder_index']['max_len']
        self.decode_loss = nn.CrossEntropyLoss(ignore_index=self.tgt_embed.end_token)
        
        # the next line used for imbalanced tree
        # self.decode_loss = nn.CrossEntropyLoss(ignore_index=self.tgt_embed.pad_token) 

        self.rk_num_neg = config['rk_num_neg']
        self.rk_margin = config['rk_margin']

        self.decode_linear = nn.Linear(config['d_model'], config['d_model'])
        self.encode_linaer = nn.Linear(config['d_model'], config['d_model'])

        self.sim_W = nn.parameter.Parameter(data=torch.randn(config['d_model'], config['d_model']), requires_grad=True)
        nn.init.xavier_normal_(self.sim_W)

        if config['rk_act'] == 'sigmoid':
            self.sim_act = nn.Sigmoid()
        elif config['rk_act'] == 'tanh':
            self.sim_act = nn.Tanh()

        # self._init_weights()

    def _init_weights(self):
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)

    def _init_prefix_mask(self, constrained_tree, device):
        # not parameters
        '''
        constrained_tree:
            an array with shape (#all nodes, vocab_size)  
                * vocab_size: all allowed next tokens, padded to the same length
                * end token: As for end token, only end token is allowed to be the next.
                * if an allowed token is the pad token, it should be ignored in generation process.
        '''
        self.constrained_prefix_tree = torch.tensor(constrained_tree, dtype=torch.long)
        self.parent_node_map = torch.zeros((self.tgt_embed.tree_node_num, 1), dtype=torch.long)

        # construct map from each node to its parent node
        branch_num = self.tgt_embed.vocab_size
        for idx, line in enumerate(self.constrained_prefix_tree):
            for i in range(branch_num):
                self.parent_node_map[line[i]] = idx

        self.constrained_prefix_tree = self.constrained_prefix_tree.to(device)
        self.parent_node_map = self.parent_node_map.to(device)

        self.temp = self.config['sm_temp'] # temperature in alignment loss

    '''
    ranking loss for the q+1 identifiers
    '''
    # @torchsnooper.snoop()
    def add_n_pair_margin_loss(self, 
                               pos_scores,
                               neg_scores,
                               pos_tgt,
                               neg_tgt
                               ):
        '''
        pos_scores: B
        neg_scores: B, num_neg
        pos_tgt: B, T
        neg_tgt: B, num_neg, T
        '''
        all_scores = torch.cat([neg_scores, pos_scores[:,None]], dim=1) # B, num_neg+1
        all_tgt = torch.cat([neg_tgt, pos_tgt[:, None]], dim=1) # B, num_neg+1, T
        all_num_same_prefix = torch.sum(all_tgt == pos_tgt[:, None], dim=2) # B, num_neg+1


        total_loss = 0

        n = all_scores.size(1)
        for i in range(1, n):
            neg = all_scores[:, :-i]
            pos = all_scores[:, i:]

            # mask neg > pos, only consider pos > neg
            same_mask = ( (all_num_same_prefix[:, i:] - all_num_same_prefix[:, :-i]) > 0).bool()
            ones = torch.ones_like(pos)

            loss = F.margin_ranking_loss(pos, neg, ones, margin=self.rk_margin*i, reduction='none')
            # loss = F.margin_ranking_loss(pos, neg, ones, margin=self.rk_margin, reduction='none')
            if same_mask.sum() > 0:
                total_loss += (loss * same_mask).sum() / same_mask.sum()

        return total_loss

    def affine_transformation(self, hid_stat, mask, is_decode_hid=True):
        '''
        hid_stat: B, L, d
        mask: B, L
        '''

        new_hid = self.decode_linear(hid_stat) if is_decode_hid else self.encode_linaer(hid_stat)
        # B, d
        new_hid = torch.sum( new_hid * mask[..., None], dim=1 ) / torch.sum(mask, dim=1, keepdim=True)

        return new_hid

    def cal_similarity(self, 
                       x, y
                       ):
        scores = self.sim_act( torch.sum( (x @ self.sim_W ) * y, dim=-1) )

        return scores


    # @torchsnooper.snoop()
    def add_node_contrast_loss(self, nodeIDs):
        '''
        add alignment loss for token embeddings
        '''

        # all_nodeIDs = nodeIDs.reshape(-1)
        unique_nodeIDs = torch.unique(input=nodeIDs)
        unique_mask1 = unique_nodeIDs == self.tgt_embed.pad_token
        unique_mask2 = unique_nodeIDs == self.tgt_embed.end_token
        unique_mask3 = unique_nodeIDs == self.tgt_embed.start_token
        unique_mask = unique_mask1 | unique_mask2 | unique_mask3

        parent_nodeIDs = F.embedding(
            input= unique_nodeIDs, weight=self.parent_node_map
        ).squeeze(dim=1)

        # mask current tokens' children tokens and special tokens
        child_of_child_nodeIDS = F.embedding(
            input=unique_nodeIDs, weight=self.constrained_prefix_tree
        ) # num_node, branch number
        aug_node_ids = unique_nodeIDs.view(1, -1).repeat(unique_nodeIDs.size(0), 1)
        child_mask = child_of_child_nodeIDS.unsqueeze(2) == aug_node_ids.unsqueeze(1)
        child_mask = torch.any(child_mask, dim=1)
        unique_mask = unique_mask | child_mask 

        child_node_emb, parent_node_emb = \
            self.tgt_embed(unique_nodeIDs), self.tgt_embed(parent_nodeIDs), \

        child_node_emb, parent_node_emb = \
            F.normalize(child_node_emb, p=2, dim=-1),\
            F.normalize(parent_node_emb, p=2, dim=-1)
        
        pos_logits = torch.sum(child_node_emb*parent_node_emb, dim=-1)
        all_logits = torch.matmul(child_node_emb, child_node_emb.T)

        # mask identical tokens
        all_logit_mask = torch.eye(n=all_logits.size(1), device=all_logits.device).bool()

        pos_scores = torch.exp(pos_logits / self.temp)
        all_scores = (~all_logit_mask) * torch.exp(all_logits / self.temp)

        softmax_value = pos_scores / (all_scores.sum(dim=-1))
        infoNCE_loss = -torch.sum((~unique_mask)*torch.log(softmax_value)) / (~unique_mask).sum()

        return infoNCE_loss

    # @torchsnooper.snoop()
    def train_step(self, batch):
        _, src, tgt, neg_tgt = batch 
        src_pad_mask = torch.where(src==0, 1, 0).bool() # B, S

        infoNCE_loss = self.add_node_contrast_loss(tgt) 

        tgt, tgt_y = tgt[:, :-1], tgt[:, 1:] # B, T
        tgt_mask = subsequent_mask(tgt.size(1)).to(tgt.device)


        encode_hidden = self.encode(src, src_pad_mask)  # B, S, d
        decode_hidden = self.decode(tgt, encode_hidden, src_pad_mask, tgt_mask) #B, T, d

        candidate_emb, candidate_mask, y_labels = self._get_next_level_tokens_for_training(tgt, tgt_y)
        logits = torch.einsum('btd,btvd->btv', [decode_hidden, candidate_emb]) # B, T, V
        logits = logits.masked_fill(candidate_mask, float('-inf'))

        decode_loss = self.decode_loss(logits.reshape(-1, logits.size(-1)), y_labels.reshape(-1))


        '''
        ranking loss
        '''

        # B, num_neg, T
        neg_tgt, neg_tgt_y = neg_tgt[:, :, :-1], neg_tgt[:, :, 1:]

        src_len, hid_dim = encode_hidden.size(1), encode_hidden.size(-1)
        encode_hidden_expand = encode_hidden[:, None].expand(-1, self.rk_num_neg, -1, -1).reshape(-1, src_len, hid_dim)
        src_pad_mask_expand = src_pad_mask[:, None].expand(-1, self.rk_num_neg, -1).reshape(-1, src_len)
        
        tgt_len = neg_tgt.size(2)
        neg_tgt_mask = neg_tgt != self.tgt_embed.end_token # B, num_neg, T
        neg_tgt = neg_tgt.reshape(-1, tgt_len)
        neg_tgt_y = neg_tgt_y.reshape(-1, tgt_len)
        neg_tgt_mask = neg_tgt_mask.reshape(-1, tgt_len)

        neg_decode_hidden = self.decode(
            neg_tgt, encode_hidden_expand, src_pad_mask_expand, tgt_mask
        )# B*num_neg, tgt, dim

        # B*num_neg, dim
        neg_decode_hidden_pooling = self.affine_transformation(neg_decode_hidden, neg_tgt_mask)
        # B, dim
        pos_decode_hidden_pooling = self.affine_transformation(decode_hidden, tgt != self.tgt_embed.end_token)

        # B, dim
        encode_hidden_pooling = self.affine_transformation(encode_hidden, ~src_pad_mask, is_decode_hid=False)

        pos_prob = self.cal_similarity(encode_hidden_pooling, pos_decode_hidden_pooling)
        neg_prob = self.cal_similarity(
            encode_hidden_pooling[:, None].expand(-1, self.rk_num_neg, -1).reshape(-1, hid_dim),
            neg_decode_hidden_pooling
        ).reshape(-1, self.rk_num_neg)

        ranking_loss = self.add_n_pair_margin_loss(
            pos_scores=pos_prob,
            neg_scores=neg_prob,
            pos_tgt=tgt,
            neg_tgt=neg_tgt.reshape(-1, self.rk_num_neg, tgt_len)
        )

        return decode_loss, ranking_loss, infoNCE_loss

    
    def predict_step(self, batch, topK):
        _, src = batch #uid not used
        src_pad_mask = torch.where(src==0, 1, 0).bool() # B, S

        candidate, scores = self.generate_topk(src, src_pad_mask, topK)

        scores = torch.exp(scores) # log_prob -> prob

        pred_items_id = torch.sum(candidate != self.tgt_embed.end_token, dim=-1, keepdim=True)
        pred_items_id = torch.gather(candidate, dim=-1, index=pred_items_id-1).squeeze(-1)
        
        return scores, pred_items_id
    
    def _get_next_level_tokens_for_training(self, 
                                            tgt: torch.tensor, 
                                            tgt_y: torch.tensor):
        '''
        Args:
            tgt (torch.tensor) of shape (batch_size, target_seq_len)
            tgt_y (torch.tensor) of shape (batch_size, target_seq_len)
        '''

        # [B, T, vocab]
        candidate_idx = F.embedding(tgt, self.constrained_prefix_tree)
        # [B, T, vocab, d]
        candidate_emb = self.tgt_embed(candidate_idx)

        # [B, T, vocab] bool
        candidate_mask = candidate_idx == self.tgt_embed.pad_token # True for invalid
        # [B, T, vocab]
        y_labels = candidate_idx == tgt_y.unsqueeze(-1)
        # [B, T]
        y_labels = torch.argmax(y_labels.float(), dim=-1)
        y_labels[tgt_y==self.tgt_embed.end_token] = self.tgt_embed.end_token # mask end token
        # y_labels[tgt_y==self.tgt_embed.pad_token] = self.tgt_embed.pad_token # mask end token. used for imbalanced tree


        return candidate_emb, candidate_mask, y_labels
 

def subsequent_mask(size):
    "Mask out subsequent positions to ignore future tokens."
    attn_shape = (size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask != 0

