import torch

from torch.utils.data import DataLoader # type: ignore
from torch.optim import Adam # type: ignore
from torch.optim.lr_scheduler import LambdaLR # type: ignore

from prepare_dataset import tokenizer, tokenize, tokenized_datasets

from model import Transformer, TransformerLoss 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if torch.cuda.is_available() else False
VOCAB_SIZE = tokenizer.vocab_size
BATCH_SIZE = 64
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
SHUFFLE = True
NUM_EPOCHS = 15
WARMUP_STEPS = 4000

def learning_rate_lambda_function(step_number):
    return D_MODEL ** (-0.5) * min((step_number + 1) ** (-0.5), (step_number + 1) * WARMUP_STEPS ** (-1.5))

def collate_function(batch):
    input_ids = torch.stack([torch.tensor(example['input_token_ids']) for example in batch])
    encoder_attention_masks = torch.stack([torch.tensor(example['encoder_attention_mask']) for example in batch])
    decoder_attention_masks = torch.stack([torch.tensor(example['decoder_attention_mask']) for example in batch])
    output_labels = torch.stack([torch.tensor(example['labels']) for example in batch])

    return {
        'input_token_ids': input_ids,
        'encoder_attention_masks': encoder_attention_masks,
        'decoder_attention_masks': decoder_attention_masks,
        'output_labels': output_labels
    }

def generate_overfit(model, encoder_inputs, encoder_padding_masks, start_token, max_length, device):
    model.eval()
    decoder_inputs = torch.tensor([start_token]).unsqueeze(0).expand(batch_size) # (batch_size, sentence_length=1)
    with torch.no_grad():
        for _ in range(max_length):
            batch_size = encoder_inputs.shape[0]
            curr_sentence_length = decoder_inputs.shape[1]
            decoder_padding_masks = torch.ones(batch_size, curr_sentence_length)
            outputs = model(encoder_inputs, decoder_inputs, encoder_padding_masks, decoder_padding_masks) # (batch_size, sentence_length, vocab_size)
            next_tokens = torch.argmax(outputs, dim=-1)[:, -1].unsqueeze(-1) # argmax returns index of largest probability, which is exactly what we want --> (batch_size, sentence_length) --> (batch_size, 1))
            decoder_inputs = torch.cat([decoder_inputs, next_tokens], dim=-1)

    return decoder_inputs


num_epochs = NUM_EPOCHS
model = Transformer(
    vocab_size=VOCAB_SIZE, 
    d_model=D_MODEL, 
    num_heads=NUM_HEADS, 
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE
)
model.to(DEVICE)
loss_function = TransformerLoss()
parameters = model.parameters()
optimizer = Adam(parameters, lr=1e-4)
scheduler = LambdaLR(optimizer, learning_rate_lambda_function)

# First overfit to small dataset to ensure that the model is working
small_loader = DataLoader(
    tokenized_datasets['train'].select(range(128)), # select first 128 examples
    batch_size=BATCH_SIZE, 
    shuffle=SHUFFLE, 
    collate_fn=collate_function, 
    pin_memory=PIN_MEMORY, 
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS
)

for epoch in range(100):
    model.train()
    print("CURRENT EPOCH: " + str(epoch))
    for batch_index, batch in enumerate(small_loader):
        optimizer.zero_grad()
        input_token_ids = batch['input_token_ids'].to(DEVICE)
        encoder_padding_masks = batch['encoder_attention_masks'].to(DEVICE)
        decoder_padding_masks = batch['decoder_attention_masks'].to(DEVICE)
        output_labels = batch['output_labels'].to(DEVICE)

        start_token_batch = torch.full((BATCH_SIZE, 1), tokenizer.bos_token_id)
        shifted_output_labels = torch.cat([start_token_batch, output_labels[:, :-1]], dim=-1)

        decoder_outputs = model(input_token_ids, shifted_output_labels, encoder_padding_masks, decoder_padding_masks)
        loss = loss_function(decoder_outputs, output_labels, tokenizer.pad_token_id)
        print("overfit batch loss: " + str(loss.item()))
        loss.backward()
        optimizer.step()
        scheduler.step()

# for batch_index, batch in enumerate(small_loader):
#     encoder_inputs = batch['input_token_ids']
#     start_token = tokenizer.bos_token_id
#     max_length = batch.shape[1]
#     device = DEVICE
#     generate_overfit(model, encoder_inputs, encoder_padding_masks, start_token, max_length, device)

