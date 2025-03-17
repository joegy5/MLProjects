import torch #type: ignore
import torch.nn as nn #type: ignore
from transformers import MarianTokenizer # type: ignore
from datasets import load_dataset # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torch.optim import Adam # type: ignore
from torch.optim.lr_scheduler import LambdaLR # type: ignore
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, batch_X):
        _, max_sentence_length, d_model = batch_X.shape
        positional_encodings = torch.arange(start=0, end=max_sentence_length, dtype=torch.float32).unsqueeze(-1).expand(-1, d_model).clone() # .expand() doesn't create new memory for the duplicated dimension, it uses shared memory --> clone it to not used shared memory
        embedding_dimensions = torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)
        div_factor = torch.tensor(10000) ** (embedding_dimensions / d_model)
        div_factor = div_factor.float()
        positional_encodings[:, 0::2] = torch.sin(positional_encodings[:, 0::2] / div_factor)
        positional_encodings[:, 1::2] = torch.cos(positional_encodings[:, 1::2] / div_factor)
        return batch_X.float().to(DEVICE) + positional_encodings.float().to(DEVICE)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.use_mask, self.num_heads, self.d_k = use_mask, num_heads, d_model // num_heads
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model, dtype=torch.float32, bias=False)
        nn.init.xavier_uniform_(self.W_Q.weight)
        self.W_K = nn.Linear(in_features=d_model, out_features=d_model, dtype=torch.float32, bias=False)
        nn.init.xavier_uniform_(self.W_K.weight)
        self.W_V = nn.Linear(in_features=d_model, out_features=d_model, dtype=torch.float32, bias=False)
        nn.init.xavier_uniform_(self.W_V.weight)
        self.W_O = nn.Linear(in_features=d_model, out_features=d_model, dtype=torch.float32, bias=False)
        nn.init.xavier_uniform_(self.W_O.weight)

    def create_mask(self, batch_size, sentence_length, padding_mask, use_attention_mask, encoder_sentence_len=None):
        if encoder_sentence_len is not None:
            padding_mask = padding_mask.unsqueeze(1).expand(-1, sentence_length, encoder_sentence_len).float().to(DEVICE)
            return (padding_mask == 0).to(DEVICE)
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, sentence_length, sentence_length).float().to(DEVICE)
        if use_attention_mask:
            causal_mask = torch.tril(torch.ones(sentence_length, sentence_length, dtype=torch.float32)).unsqueeze(0).expand(batch_size, sentence_length, sentence_length).float().to(DEVICE)
            combined_mask = torch.min(padding_mask, causal_mask).float().to(DEVICE)
        else:
            combined_mask = padding_mask.float().to(DEVICE)
        return (combined_mask == 0).to(DEVICE)

    def forward(self, batch_X, padding_mask, encoder_output=None):
        batch_X = batch_X.float()
        batch_size, sentence_length, d_model = batch_X.shape
        Q = self.W_Q(batch_X).reshape(batch_size, sentence_length, self.num_heads, self.d_k).permute(0, 2, 1, 3).float()
        if encoder_output is None:
            K = self.W_K(batch_X).reshape(batch_size, sentence_length, self.num_heads, self.d_k).permute(0, 2, 1, 3).float()
            V = self.W_V(batch_X).reshape(batch_size, sentence_length, self.num_heads, self.d_k).permute(0, 2, 1, 3).float()
        else:
            K = self.W_K(encoder_output).reshape(batch_size, encoder_output.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3).float()
            V = self.W_V(encoder_output).reshape(batch_size, encoder_output.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3).float()
        # torch.matmul() performs the matrix multiplication over the last 2 dimensions, broadcasting all the others
        enc_sentence_len = None if encoder_output is None else encoder_output.shape[1]
        mask = self.create_mask(batch_size, sentence_length, padding_mask.float(), self.use_mask, enc_sentence_len).unsqueeze(1).expand(batch_size, self.num_heads, sentence_length, enc_sentence_len if enc_sentence_len is not None else sentence_length)
        attention_scores = (torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float)))
        attention_scores = attention_scores.masked_fill(mask, float('-inf')).float()
        scaled_attention_scores = nn.functional.softmax(attention_scores, dim=-1).float()
        all_nan_rows_mask = torch.all(mask, dim=-1, keepdim=True) # (batch_size, num_heads, sentence_length, 1)
        scaled_attention_scores = scaled_attention_scores.masked_fill(all_nan_rows_mask, 0.0)
        scaled_dot_product_attention = torch.matmul(scaled_attention_scores, V).float() # shape = (batch_size, num_heads, sentence_length, d_v)
        scaled_dot_product_attention = scaled_dot_product_attention.permute(0, 2, 1, 3).reshape(batch_size, sentence_length, d_model).float() # Concatenate all the heads
        return self.W_O(scaled_dot_product_attention).to(DEVICE) # shape = (batch_size, sentence_length, d_model)


class FFN(nn.Module):
    def __init__(self, d_model, dropout_rate, activation=nn.ReLU()):
        super(FFN, self).__init__()
        self.activation_function = activation
        d_ff = d_model * 4

        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, batch_X):
        batch_X = self.dropout(self.activation(self.linear1(batch_X.float())))
        return self.linear2(batch_X.float()).to(DEVICE)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FFN(d_model=d_model, dropout_rate=dropout_rate)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model, dtype=torch.float32, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model, dtype=torch.float32, eps=1e-6)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch_X, padding_mask):
        batch_X = batch_X.float().to(DEVICE) + self.dropout(self.layer_norm1(self.mha(batch_X, padding_mask))).to(DEVICE)
        batch_X = batch_X.float().to(DEVICE) + self.dropout(self.layer_norm2(self.ffn(batch_X))).to(DEVICE)
        return batch_X.float().to(DEVICE)


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dropout_rate):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, batch_X, padding_mask):
        for encoder_layer in self.layers:
            batch_X = encoder_layer(batch_X, padding_mask).to(DEVICE)
        return self.layer_norm(batch_X).to(DEVICE)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, use_mask=True)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, dtype=torch.float32)
        self.mha2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        nn.init.xavier_uniform_(self.mha2.W_K.weight, gain=1.0)
        nn.init.xavier_uniform_(self.mha2.W_V.weight, gain=1.0)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, dtype=torch.float32)
        self.ffn = FFN(d_model=d_model, dropout_rate=dropout_rate)
        self.layernorm3 = nn.LayerNorm(normalized_shape=d_model, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, batch_X, encoder_output, decoder_padding_mask, encoder_padding_mask):
        batch_X = batch_X.float().to(DEVICE) + self.dropout(self.layernorm1(self.mha1(batch_X, decoder_padding_mask))).to(DEVICE)
        attn2_out = self.mha2(batch_X, encoder_padding_mask, encoder_output)
        # DEBUG: Log cross-attention norm for first batch of first epoch
        # if epoch == 0:
        #     print(f"batch {idx} Cross-Attention Norm: {attn2_out.norm(dim=-1).mean():.4f}")
        batch_X = batch_X.float().to(DEVICE) + self.dropout(self.layernorm2(attn2_out)).to(DEVICE)
        batch_X = batch_X.float().to(DEVICE) + self.dropout(self.layernorm3(self.ffn(batch_X))).to(DEVICE)
        return batch_X.float().to(DEVICE)


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dropout_rate):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate) for _ in range(num_layers)])

    def forward(self, batch_X, encoder_output, decoder_padding_mask, encoder_padding_mask):
        for decoder_layer in self.layers:
            batch_X = decoder_layer(batch_X.float(), encoder_output, decoder_padding_mask, encoder_padding_mask).to(DEVICE)
        return batch_X.float().to(DEVICE)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding()

        self.encoder_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=torch.float32)
        self.encoder = Encoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        self.decoder_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=torch.float32)
        self.decoder = Decoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size, dtype=torch.float32)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoder_input, shifted_decoder_input, encoder_padding_masks, decoder_padding_masks):
        embedded_encoder_input = self.encoder_embedding(encoder_input).float().to(DEVICE)
        embedded_decoder_input = self.decoder_embedding(shifted_decoder_input).float().to(DEVICE)

        encoder_output = self.encoder(
            self.positional_encoding(embedded_encoder_input),
            encoder_padding_masks
        ).float().to(DEVICE)

        decoder_output = self.decoder(
            self.positional_encoding(embedded_decoder_input),
            encoder_output,
            decoder_padding_masks,
            encoder_padding_masks
        ).float().to(DEVICE)

        logits = self.linear(decoder_output).float().to(DEVICE)
        return logits


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

        flattened_decoder_output = decoder_output.reshape(batch_size * sentence_length, vocab_size).to(DEVICE)
        flattened_target_sequences = target_sequences.reshape(batch_size * sentence_length).to(DEVICE)

        return nn.functional.cross_entropy(input=flattened_decoder_output,
                                           target=flattened_target_sequences,
                                           reduction='mean',
                                           ignore_index=padding_vocab_index).to(DEVICE)


wmt_dataset = load_dataset('iwslt2017', 'iwslt2017-fr-en')
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize(examples):
    english_examples = [example['en'] for example in examples['translation']]
    french_examples = [example['fr'] for example in examples['translation']]

    english_examples = tokenizer(english_examples, padding='max_length', truncation=True, max_length=128)
    french_examples = tokenizer(french_examples, padding='max_length', truncation=True, max_length=128)
    return {
        # all of these should have shape (batch_size, max_length)
        'input_token_ids': french_examples['input_ids'],
        'encoder_attention_mask': french_examples['attention_mask'], # mask for padded sequences
        'decoder_attention_mask': english_examples['attention_mask'],
        'labels': english_examples['input_ids']
    }

tokenized_datasets = wmt_dataset.map(tokenize, batched=True)
print(tokenizer.vocab_size)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if torch.cuda.is_available() else False
VOCAB_SIZE = tokenizer.vocab_size # add 1 because you manually added padding token, increasing vocab size
BATCH_SIZE = 32
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
NUM_WORKERS = 1
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
NUM_EPOCHS = 60
WARMUP_STEPS = 4000


def learning_rate_lambda_function(step_number):
    return (D_MODEL ** (-0.5)) * min((step_number + 1) ** (-0.5), (step_number + 1) * (WARMUP_STEPS ** (-1.5)))

def collate_function(batch):
    input_ids = torch.tensor([example['input_token_ids'] for example in batch])
    encoder_attention_masks = torch.tensor([example['encoder_attention_mask'] for example in batch])
    decoder_attention_masks = torch.tensor([example['decoder_attention_mask'] for example in batch])
    output_labels = torch.tensor([example['labels'] for example in batch])

    return {
        'input_token_ids': input_ids,
        'encoder_attention_masks': encoder_attention_masks,
        'decoder_attention_masks': decoder_attention_masks,
        'output_labels': output_labels
    }


def decode_beam_search(model, encoder_inputs, encoder_padding_masks, max_length, beam_width=5, device=DEVICE):
    """
    Beam search decoding for the transformer model.
    
    Args:
        model: Trained Transformer model.
        encoder_inputs: [batch_size, enc_seq_len] - French token IDs.
        encoder_padding_masks: [batch_size, enc_seq_len] - Encoder attention masks (1 for valid, 0 for padding).
        max_length: Maximum output sequence length.
        beam_width: Number of beams to maintain (default 5).
        device: Torch device (e.g., 'cuda' or 'cpu').
    
    Returns:
        [batch_size, seq_len] - Best decoded sequences for each example.
    """
    model.eval()
    batch_size = encoder_inputs.shape[0]
    vocab_size = model.linear.out_features  # From Transformer’s final linear layer

    # Initialize beams: [batch_size, beam_width, 1]
    decoder_inputs = torch.full((batch_size, beam_width, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
    # Beam scores: [batch_size, beam_width], start at 0 (log-probabilities)
    beam_scores = torch.zeros(batch_size, beam_width, device=device)

    # Track finished beams: [batch_size, beam_width]
    finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)

    with torch.no_grad():
        for step in range(max_length):
            # Current sequence length
            curr_len = decoder_inputs.shape[2]
            
            # Flatten for model input: [batch_size * beam_width, curr_len]
            flat_decoder_inputs = decoder_inputs.view(batch_size * beam_width, curr_len)
            flat_encoder_inputs = encoder_inputs.repeat_interleave(beam_width, dim=0)  # [batch_size * beam_width, enc_seq_len]
            flat_encoder_masks = encoder_padding_masks.repeat_interleave(beam_width, dim=0)  # [batch_size * beam_width, enc_seq_len]
            flat_decoder_masks = torch.ones(batch_size * beam_width, curr_len, device=device)  # [batch_size * beam_width, curr_len]

            # Model forward pass: [batch_size * beam_width, curr_len, vocab_size]
            outputs = model(flat_encoder_inputs, flat_decoder_inputs, flat_encoder_masks, flat_decoder_masks)
            # Logits for last position: [batch_size * beam_width, vocab_size]
            logits = outputs[:, -1, :]
            # Log probabilities: [batch_size * beam_width, vocab_size]
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            # Top beam_width candidates: [batch_size * beam_width, beam_width]
            top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)
            # Reshape: [batch_size, beam_width, beam_width]
            top_log_probs = top_log_probs.view(batch_size, beam_width, beam_width)
            top_tokens = top_tokens.view(batch_size, beam_width, beam_width)

            # Compute new scores: [batch_size, beam_width, beam_width]
            new_scores = beam_scores.unsqueeze(-1) + top_log_probs  # Add to previous beam scores
            # Flatten: [batch_size, beam_width * beam_width]
            new_scores = new_scores.view(batch_size, -1)
            new_tokens = top_tokens.view(batch_size, -1)

            # Select top beam_width overall: [batch_size, beam_width]
            beam_scores, indices = new_scores.topk(beam_width, dim=-1)
            # Beam indices: which old beam each new beam came from
            beam_idx = indices // beam_width  # [batch_size, beam_width]
            # Token indices: which new token was chosen
            token_idx = indices % beam_width  # [batch_size, beam_width]

            # Update sequences: [batch_size, beam_width, curr_len + 1]
            new_decoder_inputs = torch.zeros(batch_size, beam_width, curr_len + 1, dtype=torch.long, device=device)
            for b in range(batch_size):
                for k in range(beam_width):
                    old_beam = beam_idx[b, k]
                    new_decoder_inputs[b, k, :-1] = decoder_inputs[b, old_beam]  # Copy old sequence
                    new_decoder_inputs[b, k, -1] = top_tokens[b, old_beam, token_idx[b, k]]  # Add new token
            decoder_inputs = new_decoder_inputs

            # Update finished status
            finished = finished | (decoder_inputs[:, :, -1] == tokenizer.eos_token_id)
            if torch.all(finished):
                break

    # Select best sequence per example: [batch_size, seq_len]
    best_sequences = decoder_inputs[torch.arange(batch_size), beam_scores.argmax(dim=-1)]
    return best_sequences


def compute_bleu(model, data_loader, tokenizer, max_len, device):
    model.eval()
    hypotheses, references = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            encoder_inputs = batch['input_token_ids'].to(device)
            encoder_padding_masks = batch['encoder_attention_masks'].to(device)
            reference = batch['output_labels'].long().to(device)
            prediction_ids = decode_beam_search(model, encoder_inputs, encoder_padding_masks, max_len, device=device)

            special_tokens = [tokenizer.pad_token_id, tokenizer.eos_token_id]
            predicted_tokens = [token.item() for token in prediction_ids[0] if token not in special_tokens]
            reference_tokens = [token.item() for token in reference[0] if token not in special_tokens]
            if i < 3:
                print(f"Example {i}:")
                print("Input (French):", tokenizer.decode(encoder_inputs[0].tolist(), skip_special_tokens=True))
                print("Predicted:", tokenizer.decode(predicted_tokens))
                print("Reference:", tokenizer.decode(reference_tokens))
            hypotheses.append(predicted_tokens)
            references.append([reference_tokens])

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    return bleu_score * 100


num_epochs = NUM_EPOCHS
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=512,
    num_heads=8,
    num_layers=6,
    dropout_rate=0.1
)

model.to(DEVICE)
model.float() # convert all model parameters to torch.float32
loss_function = TransformerLoss()
parameters = model.parameters()
optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=learning_rate_lambda_function)

train_loader = DataLoader(
    tokenized_datasets['train'],
    batch_size=32,
    shuffle=True,
    collate_fn=collate_function,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS
)
train_bleu_loader = DataLoader(
    tokenized_datasets['train'].select(range(1000)),
    batch_size=1,
    shuffle=True,
    collate_fn=collate_function,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS
)
validation_loader = DataLoader(
    tokenized_datasets['validation'], 
    batch_size=1, 
    shuffle=False, 
    collate_fn=collate_function, 
    pin_memory=PIN_MEMORY, 
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS
)
test_loader = DataLoader(
    tokenized_datasets['test'], 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_function, 
    pin_memory=PIN_MEMORY, 
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS
)

print(f"Training set size: {len(train_loader.dataset)}")
print(f"Validation set size: {len(validation_loader.dataset)}")
print(f"Test set size: {len(test_loader.dataset)}")

# MIN_GRAD_NORM = 1e-5
best_val_bleu = float('-inf')
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    print("CURRENT EPOCH: " + str(epoch))
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Move everything to device and set correct dtypes
        input_token_ids = batch['input_token_ids'].long().to(DEVICE)
        encoder_padding_masks = batch['encoder_attention_masks'].to(DEVICE)
        decoder_padding_masks = batch['decoder_attention_masks'].to(DEVICE)
        output_labels = batch['output_labels'].long().to(DEVICE)

        # Create start tokens
        start_token_batch = torch.full((input_token_ids.size(0), 1),
                                     tokenizer.eos_token_id,
                                     dtype=torch.long,
                                     device=DEVICE)
        shifted_output_labels = torch.cat([start_token_batch, output_labels[:, :-1]], dim=-1).to(DEVICE)

        decoder_outputs = model(input_token_ids, shifted_output_labels, encoder_padding_masks, decoder_padding_masks).to(DEVICE)
        # Compute loss
        loss = loss_function(decoder_outputs, output_labels, tokenizer.pad_token_id)
        epoch_loss += loss.item() * BATCH_SIZE
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    print("TRAINING LOSS: " + str(epoch_loss / len(train_loader.dataset)))
    if epoch % 5 == 0:
        training_bleu = compute_bleu(model, train_bleu_loader, tokenizer, max_len=128, device=DEVICE)
        print(f"TRAINING BLEU: {training_bleu}")
        val_bleu = compute_bleu(model, validation_loader, tokenizer, max_len=128, device=DEVICE)
        print(f"VALIDATION BLEU: {val_bleu}")

        if val_bleu > best_val_bleu: 
            torch.save(model.state_dict(), "best_transformer_weights.pth")
            best_val_bleu = val_bleu

print("BEST VALIDATION_BLEU: " + str(best_val_bleu))

test_bleu = compute_bleu(model, test_loader, tokenizer, max_len=128, device=DEVICE)
print(f"TEST BLEU: {test_bleu}")
