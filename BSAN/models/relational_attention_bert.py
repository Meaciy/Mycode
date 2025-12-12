import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import random



#实现了层归一化，其作用是对输入进行均值和标准差的归一化，然后通过可学习的缩放和偏移参数进行线性变换
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class TransformerLayer(nn.Module): 
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio,seed): 
        super().__init__() 
        self.seed=seed
        self.hidden_dim = hidden_dim 
        self.attention_heads = attention_heads 
        self.attn_dropout = nn.Dropout(attn_dropout_ratio) 

        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 

        self.linear_attn_out = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
            ) 
        norm_class = nn.LayerNorm 
        self.MY=nn.Linear(100,768)
        self.norm1 = norm_class(hidden_dim) 
        self.norm2 = norm_class(hidden_dim)     
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
        ) 
    
    def forward(self, x, src_mask): # x shape (B x L x F) 
        seed=self.seed
        q = self.linear_q(x) 
        k = self.linear_k(x) 
        v = self.linear_v(x) 

        dim_split = self.hidden_dim // self.attention_heads 
        q_heads = torch.cat(q.split(dim_split, 2), dim=0) 
        k_heads = torch.cat(k.split(dim_split, 2), dim=0) 
        v_heads = torch.cat(v.split(dim_split, 2), dim=0) 
        
        attention_score = q_heads.bmm(k_heads.transpose(1, 2)) # (B x H, L, L) 

        attention_score = attention_score / math.sqrt(self.hidden_dim // self.attention_heads) 

        inf_mask = (~src_mask).unsqueeze(1).to(dtype=torch.float) * -1e9 # (B, 1, L) 
        inf_mask = torch.cat([inf_mask for _ in range(self.attention_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 

        A = self.attn_dropout(A) 
        attn_out = torch.cat((A.bmm(v_heads)).split(q.size(0), 0), 2) # (B, L, F) 

        attn_out = self.linear_attn_out(attn_out) 
        attn_out = attn_out + x 
        attn_out = self.norm1(attn_out) 
        
        out = self.ffn(attn_out) + attn_out 

        out = self.norm2(out) 

        return out 

class TextGINConv(nn.Module): 
    def __init__(self, hidden_dim, dropout_ratio, seed, edge_dim: int = None): 
        super().__init__() 
        self.seed=seed
        self.linear_e = nn.Linear(edge_dim, hidden_dim) 
        norm_class = nn.LayerNorm 
        self.ffn = nn.Sequential( 
                        nn.Linear(hidden_dim, 2 * hidden_dim), 
                        nn.ReLU(), 
                        nn.Linear(2 * hidden_dim, hidden_dim))
    def forward(self, x, adj, e): # x is (B, L, F), adj is (B, L, L), e is (B, L, L, Fe) 
        seed=self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        e = self.linear_e(e) # (B, L, L, F) 
        m = (x.unsqueeze(1) + e).relu() # (B, L, L, F) 
        z = torch.zeros_like(m) 
        m = torch.where((adj != 0).unsqueeze(-1), m, z) # (B, L, L, F) 
        out = m.sum(dim=-2) # (B, L, F) 
        out = self.ffn(out + x) 
        return out 

def attention(aspect,query, key,bias_m, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #aspect_scores = torch.tanh(torch.add(torch.matmul(aspect, key.transpose(-2, -1)), bias_m))  # aspect_scores [8, 5, 100, 100]
    #scores=torch.add(scores, aspect_scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(d_model, self.d_k)
        self.bias_m = nn.Parameter(torch.Tensor(1))

    def forward(self,aspect, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)    
        aspect = self.dense(aspect) 
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn = attention(aspect,query, key,self.bias_m, mask=mask, dropout=self.dropout)
        return attn


#这个模型结合了 BERT 模型、关系注意力和 Evoformer 结构
class RelationalAttentionBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.num_layers
        self.bert = bert
        self.attention_heads = opt.attention_heads
        self.hidden_dim = opt.bert_dim // 2
        # Bert 模型相关的组件
        self.attdim = 100
        self.bert_dim = opt.bert_dim
        self.bert_layernorm = LayerNorm(opt.bert_dim)
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        
        self.attn = MultiHeadAttention(self.attention_heads, self.bert_dim)
        self.Wxx = nn.Linear(self.bert_dim, 100)
        self.W = nn.Linear(self.attdim,self.attdim)
        self.Wx= nn.Linear(self.attention_heads+self.attdim*2, self.attention_heads)
        self.BB=nn.Linear(self.hidden_dim,self.bert_dim)
        self.aggregate_W =nn.Linear(self.attdim,self.bert_dim)
        # 边嵌入相关的组件
        self.edge_emb = torch.load(opt.amr_edge_pt) \
            if opt.edge == "normal" or opt.edge == "same" else nn.Embedding(56000, 1024)
        self.edge_emb_layernorm = nn.LayerNorm(opt.amr_edge_dim)
        self.edge_emb_drop = nn.Dropout(opt.edge_dropout)
        self.edge_dim_change = nn.Linear(opt.amr_edge_dim, self.hidden_dim, bias=False)
        self.AA=nn.Linear(100,self.bert_dim)
       #####  
        self.linear_in = nn.Linear(opt.bert_dim , opt.hidden_dim) 
        self.W1=nn.Linear(self.hidden_dim, opt.deprel_dim )
        self.linear_out = nn.Linear(opt.hidden_dim + opt.bert_dim, opt.polarities_dim) 
        self.bert_drop = nn.Dropout(opt.bert_dropout) 
        self.pooled_drop = nn.Dropout(opt.bert_dropout) 
        self.ffn_dropout = opt.ffn_dropout 
        self.graph_convs = nn.ModuleList() 
        self.norms = nn.ModuleList() 
        self.transformer_layers = nn.ModuleList() 
        norm_class = nn.LayerNorm        
        for i in range(opt.num_layers): 
            graph_conv = TextGINConv(opt.hidden_dim, 
                            dropout_ratio=opt.ffn_dropout, 
                            edge_dim=opt.deprel_dim,
                            seed=opt.seed) 
            
            self.graph_convs.append(graph_conv) 
            self.norms.append(norm_class(opt.hidden_dim)) 
            self.transformer_layers.append(TransformerLayer( 
                                    opt.hidden_dim, 
                                    opt.attention_heads, 
                                    attn_dropout_ratio=opt.attn_dropout, 
                                    ffn_dropout_ratio=opt.ffn_dropout,
                                    seed=opt.seed )) 
        self.affine1 = nn.Parameter(torch.Tensor(self.bert_dim, self.bert_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.bert_dim, self.bert_dim))
        # 最终层的组件
        self.final_dropout = nn.Dropout(opt.final_dropout)
        self.final_layernorm = LayerNorm(opt.bert_dim)
        # 分类器

        self.classifier = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
  

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, edge_adj, src_mask, aspect_mask= inputs
        # BERT模型的输出
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)  # sequence_output.shape [8, 100, 768] 
        sequence_output = self.bert_layernorm(sequence_output)

        # sequence_output.shape [8, 100, 768]

        mask = src_mask.unsqueeze(-2) 
        token_inputs = self.bert_drop(sequence_output)   # token_inputs.shape [8, 100, 768]
        pooled_output = self.pooled_drop(pooled_output)
        
        # 边嵌入的处理
        batch_size, max_length, _ = edge_adj.size()
        edge_adj = self.edge_emb(edge_adj)
        edge_adj = self.edge_emb_layernorm(edge_adj)
        edge_adj = self.edge_emb_drop(edge_adj)   
        
        #  edge_adj.shape [8, 100, 100, 1024]
        edge_adj = self.edge_dim_change(edge_adj)
        #  edge_adj.shape [8, 100, 100, 384]
        e=self.W1(edge_adj)
        edge_adj=edge_adj.mean(dim=1)
        edge_adj=self.BB(edge_adj)
         # 计算与方面词相关的输出
        inputs = self.Wxx(token_inputs)
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask_repeat = aspect_mask.unsqueeze(-1)
        batch = src_mask.size(0)
        aspectmask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100) 
        aspect = (inputs*aspectmask).sum(dim=1) / asp_wn
        aspect=self.AA(aspect)
        attn_tensor = self.attn(aspect,edge_adj, edge_adj, mask)
        weight_adj=attn_tensor   
        gcn_outputs=inputs  # gcn_outputs [8, 100, 100]
        layer_list = [inputs]
     
        #实现了一个多层的GCN结构，其中每一层都执行图卷积操作、线性变换、非线性激活、dropout以及邻接矩阵的更新
        for i in range(self.layers):
            #图卷积操作：
            gcn_outputs = gcn_outputs.unsqueeze(1).expand(batch, self.attention_heads, self.attdim, self.attdim)   
            # gcn_outputs [8, 5, 100, 100]
            Ax = torch.matmul(weight_adj, gcn_outputs)     # Ax  [8, 5, 100, 100]
            Ax = Ax.mean(dim=1)  # Ax [8, 100, 100]
            #线性变换和非线性激活
            Ax = self.W(Ax)      # Ax [8, 100, 100]
            weights_gcn_outputs = F.relu(Ax)  # weights_gcn_outputs [8, 100, 100]
            #更新GCN输出和层列表
            gcn_outputs = weights_gcn_outputs  # gcn_outputs [8, 100, 100]
            layer_list.append(gcn_outputs)
            gcn_outputs = self.gcn_drop(gcn_outputs) if i < self.layers - 1 else gcn_outputs  
            # gcn_outputs [8, 100, 100]
            #更新邻接矩阵
            weight_adj=weight_adj.permute(0, 2, 3, 1).contiguous()   #[8, 100, 100, 5]
            node_outputs1 = gcn_outputs.unsqueeze(1).expand(batch, self.attdim, self.attdim, self.attdim)    #[8, 100, 100, 100]
            node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()   #[8, 100, 100, 100]
            node = torch.cat([node_outputs1, node_outputs2], dim=-1)  #[8, 100, 100, 200]
            edge_n=torch.cat([weight_adj, node], dim=-1)  #[8, 100, 100, 205]
        
            edge = self.Wx(edge_n)  #[8, 100, 100, 5]
            edge = self.gcn_drop(edge) if i < self.layers - 1 else edge #[8, 100, 100, 5]
            weight_adj=edge.permute(0,3,1,2).contiguous()  # weight_adj [8, 5, 100, 100]



        outputs = torch.cat(layer_list, dim=-1)
        # node_outputs=self.Wi(gcn_outputs)
        #node_outputs=self.aggregate_W(outputs)
        node_outputs=F.relu(gcn_outputs) 
        tokens=self.aggregate_W (node_outputs)
        
        
        h = self.linear_in(token_inputs) 

        for i in range(self.layers): 
            h0 = h 
            # Graph Conv 
            h = self.graph_convs[i](h, adj_dep, e) 
            # Middle layer 
            h = self.norms[i](h) 
            h = h.relu() 
            h = F.dropout(h, self.ffn_dropout, training=self.training)   
            # Transformer Layer 
            h = self.transformer_layers[i](h, src_mask) 
            # Skip connection or Jumping Knowledge 
            h = h + h0 #16.100.768
        ###
        for l in range(self.layers):
            A1 = F.softmax(torch.bmm(torch.matmul(tokens, self.affine1), torch.transpose(h, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(h, self.affine2), torch.transpose(tokens, 1, 2)), dim=-1)
            tokens, gAxW_ag = torch.bmm(A1,h), torch.bmm(A2,tokens)
            tokens = self.gcn_drop(tokens) if l < self.layers - 1 else tokens
            h = self.gcn_drop(h) if l < self.layers - 1 else h

        outputs1 = (tokens*aspect_mask_repeat).sum(dim=1) / asp_wn
        outputs2=(h * aspect_mask_repeat).sum(dim=1) / asp_wn
        # 构建最终输入
        final_outputs = torch.cat((outputs1, outputs2,pooled_output), dim=-1)
        # 通过分类器获得最终预测结果
        logits = self.classifier(final_outputs)  # final_outputs.shape [8, 1536]        
        #logits.shape [8, 3]       
        return logits, None