import torch #type: ignore
import torch.nn as nn #type: ignore
import numpy #type: ignore


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, batch_X):
        _, max_sentence_length, d_model = batch_X.shape
        
        positional_encodings = torch.arange(start=0, end=max_sentence_length).unsqueeze(-1).expand(-1, d_model).clone() # .expand() doesn't create new memory for the duplicated dimension, it uses shared memory --> clone it to not used shared memory
        embedding_dimensions = torch.arange(start=0, end=d_model, step=2).unsqueeze(-1).expand(-1, d_model).clone()
        positional_encodings[:, 0::2] = torch.sin(positional_encodings[:, 0::2] / (10000 ** (embedding_dimensions / d_model))).unsqueeze(-1).expand(-1, d_model).clone()
        positional_encodings[:, 1::2] = torch.cos(positional_encodings[:, 1::2] / (10000 ** (embedding_dimensions / d_model)))

        return batch_X + positional_encodings # broadcasting so that positional_encodings added to every training example


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_mask=False):
        super(MultiHeadAttention, self).__init__()
        # Broadcasting --> don't need to worry about batch dimension
    
        # nn.Linear matrix has shape (out_features, in_features), performs computation XW
        self.use_mask = use_mask
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        self.d_v = self.d_k
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_O = nn.Linear(in_features=d_model, out_features=d_model)

    def create_mask(self, batch_size, sentence_length, padding_mask, use_attention_mask):
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, sentence_length, sentence_length)
        if use_attention_mask:
            causal_mask = torch.tril(torch.ones(sentence_length, sentence_length)).unsqueeze(0).expand(batch_size, sentence_length, sentence_length)
            combined_mask = torch.min(padding_mask, causal_mask)
        else:
            combined_mask = padding_mask
        return combined_mask == 0

    def forward(self, batch_X, padding_mask, dropout_rate, encoder_output=None):
        batch_size, sentence_length, d_model = batch_X.shape
        
        Q = None
        if encoder_output is not None:
            Q = self.W_Q(encoder_output).permute(0, 2, 1).reshape(batch_size, sentence_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        else:
            Q = self.W_Q(batch_X).permute(0, 2, 1).reshape(batch_size, sentence_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_K(batch_X).permute(0, 2, 1).reshape(batch_size, sentence_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(batch_X).permute(0, 2, 1).reshape(batch_size, sentence_length, self.num_heads, self.d_v).permute(0, 2, 1, 3)

        # torch.matmul() performs the matrix multiplication over the last 2 dimensions, broadcasting all the others
        mask = self.create_mask(batch_size, sentence_length, padding_mask, self.use_mask).unsqueeze(1).expand(batch_size, self.num_heads, sentence_length, sentence_length)
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float)).masked_fill(mask, float('-inf'))
        
        scaled_attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        #scaled_dot_product_attention = nn.Dropout(dropout_rate)(torch.matmul(scaled_attention_scores, V)) # shape = (batch_size, num_heads, sentence_length, d_v)
        scaled_dot_product_attention = torch.matmul(scaled_attention_scores, V) # shape = (batch_size, num_heads, sentence_length, d_v)
        
        # Concatenate all the heads
        scaled_dot_product_attention = scaled_dot_product_attention.permute(0, 2, 1, 3).reshape(batch_size, sentence_length, d_model)
        
        return self.W_O(scaled_dot_product_attention) # shape = (batch_size, sentence_length, d_model)
        
class FFN(nn.Module):
    def __init__(self, d_model, activation=nn.ReLU()):
        super(FFN, self).__init__()
        d_ff = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            activation,
            nn.Linear(in_features=d_ff, out_features=d_model)
        )

    def forward(self, batch_X):
        return self.ffn(batch_X)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FFN(d_model=d_model)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, batch_X, padding_mask, dropout_rate):
        # print("input shape pre mha: " + str(batch_X.shape))
        batch_X = self.layer_norm1(batch_X + self.mha(batch_X, padding_mask, dropout_rate))
        # print("input post mha: " + str(batch_X.shape))
        #batch_X = self.layer_norm2(nn.Dropout(dropout_rate)(batch_X + self.ffn(batch_X)))
        batch_X = self.layer_norm2(batch_X + self.ffn(batch_X))
        return batch_X
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        layer_list= [EncoderLayer(d_model=d_model, num_heads=num_heads) for l in range(num_layers)]
        for l in range(num_layers):
            self.add_module(f"EncoderLayer{l}", layer_list[l])

    def forward(self, batch_X, padding_mask, dropout_rate):
        for l in range(self.num_layers):
            batch_X = self._modules[f"EncoderLayer{l}"](batch_X, padding_mask, dropout_rate)
        return batch_X
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, use_mask=True)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model)
        self.ffn = FFN(d_model=d_model)
        self.layernorm3 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, batch_X, encoder_output, padding_mask, dropout_rate):
        batch_X = self.layernorm1(batch_X + self.mha1(batch_X, padding_mask, dropout_rate))
        batch_X = self.layernorm2(batch_X + self.mha2(batch_X, padding_mask, dropout_rate, encoder_output))
        
        #batch_X = self.layernorm3(nn.Dropout(dropout_rate)(batch_X + self.ffn(batch_X)))
        batch_X = self.layernorm3(batch_X + self.ffn(batch_X))
        return batch_X

    

class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        layer_list = [DecoderLayer(d_model=d_model, num_heads=num_heads) for l in range(num_layers)]
        for l in range(num_layers):
            self.add_module(f"DecoderLayer{l}", layer_list[l])

    def forward(self, batch_X, encoder_output, padding_mask, dropout_rate):
        for l in range(self.num_layers):
            batch_X = self._modules[f"DecoderLayer{l}"](batch_X, encoder_output, padding_mask, dropout_rate)
        return batch_X
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate):
        super(Transformer, self).__init__()
        self.vocab_size=vocab_size
        self.dropout_rate = dropout_rate
        self.positional_encoding = PositionalEncoding()
        self.encoder_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.encoder_embedding_dropout = nn.Dropout(dropout_rate)
        self.encoder = Encoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads)
        self.decoder_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.decoder_embedding_dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_input, shifted_decoder_input, encoder_padding_masks, decoder_padding_masks):
        # embedded_encoder_input = self.encoder_embedding_dropout(self.encoder_embedding(encoder_input))
        embedded_encoder_input = self.encoder_embedding(encoder_input)
        # print("embedded encoder input: " + str(embedded_encoder_input))
        # embedded_decoder_input = self.decoder_embedding_dropout(self.decoder_embedding(shifted_decoder_input))
        embedded_decoder_input = self.decoder_embedding(shifted_decoder_input)
        # print("embedded decoder input: " + str(embedded_decoder_input))
        encoder_output = self.positional_encoding(embedded_encoder_input)
        # print("encoder positional_encodings: " + str(encoder_output))
        encoder_output = self.encoder(encoder_output, encoder_padding_masks, self.dropout_rate)
        # print("encoder output: " + str(encoder_output))
        decoder_output = self.decoder(self.positional_encoding(embedded_decoder_input), encoder_output, decoder_padding_masks, self.dropout_rate)
        # print("decoder_output: " + str(decoder_output))
        output_probabilities = self.softmax(self.decoder_dropout(self.linear(decoder_output)))
        # print("output probabilities: " + str(output_probabilities))
        return output_probabilities

    

class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()

    def forward(self, decoder_output, target_sequences, padding_vocab_index):
        # decoder_output has shape (batch_size, sentence_length, vocab_size)
        # target_sequences has shape (batch_size, sentence_length)
        # for each training example, each of the vocab_size positions in each row
            # has a corresponding probability of being selected, and each corresponding row in the target
            # will have a value equal to the correct position representing a word in the vocabulary
        # print(target_sequences)
        batch_size, sentence_length, vocab_size = decoder_output.shape
        flattened_decoder_output = decoder_output.reshape(batch_size * sentence_length, vocab_size)
        flattened_target_sequences = target_sequences.reshape(batch_size * sentence_length)
        
        return nn.functional.cross_entropy(input=flattened_decoder_output, 
                                           target=flattened_target_sequences, 
                                           reduction='mean',
                                           ignore_index=padding_vocab_index)
