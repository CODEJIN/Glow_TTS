import torch
import numpy as np
import yaml, logging, math

class RPR_Multihead_Attention(torch.nn.Module):
    def __init__(
        self,
        query_channels,
        calc_channels,
        out_channels,
        num_heads,
        relative_postion_clipping_distance= None,
        share_relative_postion_weight= True,
        proximal_bias= False,
        block_mask_length= None,
        dropout_rate= 0.0,
        key_channels= None,
        value_channels= None,
        ):
        assert calc_channels % num_heads == 0, 'calc_channels must be dividable by num_heads.'

        super(RPR_Multihead_Attention, self).__init__()
        self.num_heads = num_heads
        self.calc_channels_per_head = calc_channels // num_heads
        self.relative_postion_clipping_distance = relative_postion_clipping_distance
        self.proximal_bias = proximal_bias
        self.block_mask_length = block_mask_length

        self.layer_Dict = torch.nn.ModuleDict()        
        self.layer_Dict['Query'] = torch.nn.Conv1d(
            in_channels= query_channels,
            out_channels= calc_channels,
            kernel_size= 1
            )
        self.layer_Dict['Key'] = torch.nn.Conv1d(
            in_channels= key_channels or query_channels,
            out_channels= calc_channels,
            kernel_size= 1
            )
        self.layer_Dict['Value'] = torch.nn.Conv1d(
            in_channels= value_channels or key_channels or query_channels,
            out_channels= calc_channels,
            kernel_size= 1
            )
        torch.nn.init.xavier_uniform_(self.layer_Dict['Query'].weight)
        torch.nn.init.xavier_uniform_(self.layer_Dict['Key'].weight)
        torch.nn.init.xavier_uniform_(self.layer_Dict['Value'].weight)

        self.layer_Dict['Projection'] = torch.nn.Conv1d(
            in_channels= calc_channels,
            out_channels= out_channels,
            kernel_size= 1
            )
        
        self.layer_Dict['Dropout'] = torch.nn.Dropout(
            p= dropout_rate
            )

        if not relative_postion_clipping_distance is None:
            num_Head_Weight = 1 if share_relative_postion_weight else num_heasds
            weight_STD = self.calc_channels_per_head ** -0.5
            self.weight_K = torch.nn.Parameter(
                torch.randn(num_Head_Weight, relative_postion_clipping_distance * 2 + 1, self.calc_channels_per_head) * weight_STD
                )
            self.weight_V = torch.nn.Parameter(
                torch.randn(num_Head_Weight, relative_postion_clipping_distance * 2 + 1, self.calc_channels_per_head) * weight_STD
                )

    def forward(self, queries, keys= None, values= None, masks= None):
        '''
        if keys and values are None, queries == values == keys.
        else if key or values are None, values == keys.
        When value != key, the time step must be same between keys and values.
        '''
        assert self.relative_postion_clipping_distance is None or (keys is None and values is None), 'Relative position is for self-attention.'
        assert not self.proximal_bias or (keys is None and values is None), 'Proximal bias is for self-attention.'
        assert self.block_mask_length is None or (keys is None and values is None), 'Block mask is for self-attention.'

        keys = keys if not keys is None else values if not values is None else queries
        values = values or keys

        queries = self.layer_Dict['Query'](queries)
        keys = self.layer_Dict['Key'](keys)
        values = self.layer_Dict['Value'](values)

        attentions, alignments = self.Calc_Attention(
            queries= queries,
            keys= keys,
            values= values,
            masks= masks
            )

        return self.layer_Dict['Projection'](attentions), alignments

    def Calc_Attention(self, queries, keys, values, masks):
        batches, channels, queries_Time =  queries.size()
        keys_Time = keys.size(2)

        queries = queries.view(batches, self.num_heads, self.calc_channels_per_head, queries_Time).transpose(2, 3)   # [Batch, Head, Query_t, Channel // Head]
        keys = keys.view(batches, self.num_heads, self.calc_channels_per_head, keys_Time).transpose(2, 3)    # [Batch, Head, Key_t, Channel // Head]
        values = values.view(batches, self.num_heads, self.calc_channels_per_head, keys_Time).transpose(2, 3)    # [Batch, Head, Key_t, Channel // Head]

        scores = queries @ keys.transpose(3, 2) / math.sqrt(self.calc_channels_per_head)    # [Batch, Head, Query_t, Channel // Head] @ [Batch, Head, Channel // Head, Key_t] -> @ [Batch, Head, Query_t, Key_t]

        if not self.relative_postion_clipping_distance is None: # Because this is for self-attention, Time == Key_t == Query_t
            relative_Position_Key_Embedding = self.Get_Relative_Embedding(relative_embeddings= self.weight_K, length= keys_Time)    #[1(Head), Time * 2 - 1, Channel // Head]
            positions = queries @ relative_Position_Key_Embedding.unsqueeze(0).transpose(3, 2)    # [Batch, Head, Time, Channel // Head] @ [1, 1(Head), Channel // Head, Time * 2 - 1] -> [Batch, Head, Time, Time * 2 - 1]
            positions = self.Relative_Position_to_Absolute_Position(positions)
            scores += positions / math.sqrt(self.calc_channels_per_head)
        
        if self.proximal_bias:
            scores += self.Get_Proximal_Bias(length= keys_Time)

        if not masks is None:
            if not self.block_mask_length is None:
                masks *= torch.ones_like(scores).triu(-self.block_mask_length).tril(self.block_mask_length)
            scores = scores.masked_fill(masks == 0, -1e+4)

        alignments = torch.nn.functional.softmax(scores, dim= -1)    # [Batch, Head, Query_t, Key_t]
        alignments = self.layer_Dict['Dropout'](alignments)
        attensions = alignments @ values    # [Batch, Head, Query_t, Key_t] @ [Batch, Head, Key_t, Channel // Head] -> [Batch, Head, Query_t, Channel // Head]

        if not self.relative_postion_clipping_distance is None: # Because this is for self-attention, Time == Key_t == Query_t
            positions = self.Absolute_Position_to_Relative_Position(alignments)  # [Batch, Head, Time, Time * 2 - 1]
            relative_Position_Value_Embedding = self.Get_Relative_Embedding(relative_embeddings= self.weight_V, length= keys_Time)    #[1(Head), Time * 2 - 1, Channel // Head]
            attensions += positions @ relative_Position_Value_Embedding.unsqueeze(0)    # [Batch, Head, Time, Time * 2 - 1] @ [1, 1(Head), Time * 2 - 1, Channel // Head] -> [Batch, Head, Time, Channel // Head]
        
        return attensions.transpose(3, 2).contiguous().view(batches, channels, queries_Time), alignments


    def Get_Relative_Embedding(self, relative_embeddings, length):
        embedding_Length = 2 * length - 1

        pads = max(embedding_Length - self.relative_postion_clipping_distance * 2 - 1, 0) / 2
        relative_embeddings = Pad(
            relative_embeddings,
            [[0, 0], [math.ceil(pads), math.floor(pads)], [0, 0]]
            )
        index = max(self.relative_postion_clipping_distance + 1 - length, 0)
        return relative_embeddings[:, index: index + embedding_Length]

    def Relative_Position_to_Absolute_Position(self, x):
        batches, heads, length, _ = x.size()
        
        x = Pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])    # [Batch, Head, Time, Time * 2]
        x = x.view(batches, heads, length * length * 2)     # [Batch, Head, Time * Time * 2]
        x = Pad(x, [[0, 0], [0, 0], [0, length - 1]])   # [Batch, Head, Time * Time * 2 + Time - 1]
        x = x.view(batches, heads, length + 1, length * 2 - 1)  # [Batch, Head, Time + 1, Time * 2 - 1]
        
        return x[:, :, :length, length - 1:]    # [Batch, Head, Time, Time]

    def Get_Proximal_Bias(self, length):
        sequence = torch.arange(length, dtype= torch.float32)   # [Time]
        difference = sequence.unsqueeze(0) - sequence.unsqueeze(1)  # [Time, Time]
        return -torch.log1p(torch.abs(difference)).unsqueeze(0).unsqueeze(0)   # [1, 1, Time, Time]

    def Absolute_Position_to_Relative_Position(self, x):
        batches, heads, length, _ = x.size()    # Because this is for self-attention, Time == Key_t == Query_t
        
        x = Pad(x, [[0, 0], [0, 0], [0, 0], [0, length - 1]])    # [Batch, Head, Time, Time * 2 - 1]
        x = x.view(batches, heads, length * (length * 2 - 1))     # [Batch, Head, Time * (Time * 2 - 1)]
        x = Pad(x, [[0, 0], [0, 0], [length, 0]])   # [Batch, Head, Time * Time * 2]
        x = x.view(batches, heads, length, length * 2)  # [Batch, Head, Time, Time * 2]
        
        return x[:, :, :, 1:]    # [Batch, Head, Time, Time * 2 - 1]


def Pad(x, pad, mode='constant', value= 0):
    return torch.nn.functional.pad(
        input= x,
        pad= [size for sizes in pad[::-1] for size in sizes],
        mode= mode,
        value= value
        )
    



