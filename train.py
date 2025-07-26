import json
import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import GradScaler, autocast # Import for mixed precision

# --- Configuration ---
# Updated file paths to match your local setup
TOKENIZED_DATA_PATH = "data/tokenize_data.jsonl"
TOKENIZER_PATH = "data/tokenizer.json" 
MODEL_SAVE_PATH = "./models" # Directory to save models
BATCH_SIZE = 4
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
LEARNING_RATE = 0.0001
NUM_EPOCHS = 3 # Increased for demonstration
LOG_INTERVAL = 1
# Increased max_len to handle long sequences in your data
MAX_SEQ_LEN = 20000 

# --- Step 1: Streaming Dataset Class (using an IterableDataset) ---

class SummarizationDataset(IterableDataset):
    """
    An iterable-style dataset that streams data from a JSONL file.
    """
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def __iter__(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at '{self.file_path}'")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield {
                        "article": torch.tensor(data["article_ids"], dtype=torch.long),
                        "abstract": torch.tensor(data["abstract_ids"], dtype=torch.long)
                    }
                except (json.JSONDecodeError, KeyError):
                    print(f"Warning: Skipping malformed line: {line.strip()}")


# --- Step 2: Collate Function for Padding ---

def collate_batch(batch, pad_id):
    """
    Pads sequences in a batch to the same length.
    """
    articles = [item['article'] for item in batch]
    abstracts = [item['abstract'] for item in batch]

    padded_articles = nn.utils.rnn.pad_sequence(articles, batch_first=True, padding_value=pad_id)
    padded_abstracts = nn.utils.rnn.pad_sequence(abstracts, batch_first=True, padding_value=pad_id)

    return {"article": padded_articles, "abstract": padded_abstracts}


# --- Step 3: Transformer Model for Summarization ---

class PositionalEncoding(nn.Module):
    # Updated to accept a dynamic max_len
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape is (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]

class SummarizationTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        # Permute src and tgt to (seq_len, batch_size, d_model) for positional encoding
        src_embedded = self.embedding(src) * torch.sqrt(torch.tensor(D_MODEL, dtype=torch.float32))
        src_pos = self.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2)

        tgt_embedded = self.embedding(tgt) * torch.sqrt(torch.tensor(D_MODEL, dtype=torch.float32))
        tgt_pos = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)

        output = self.transformer(
            src_pos, tgt_pos,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.fc_out(output)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        return nn.Transformer.generate_square_subsequent_mask(sz).to(device)


# --- Step 4: Training Function ---

def train_model(model, dataloader, vocab_size, pad_id, epochs, learning_rate, save_path):
    """
    The main training loop, now with checkpointing to resume training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    os.makedirs(save_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    start_epoch = 0

    # --- Check for the latest checkpoint to resume training ---
    latest_checkpoint_path = os.path.join(save_path, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    model.train()

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        num_batches = 0
        print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")

        for i, batch in enumerate(dataloader):
            src_batch = batch['article'].to(device)
            tgt_batch = batch['abstract'].to(device)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            src_padding_mask = (src_batch == pad_id).to(device)
            tgt_padding_mask = (tgt_input == pad_id).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1), device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                output = model(
                    src=src_batch,
                    tgt=tgt_input,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    tgt_mask=tgt_mask
                )
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if (i + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / num_batches
        print(f"--- End of Epoch {epoch+1}, Average Loss: {avg_epoch_loss:.4f} ---")

        # --- Save a complete checkpoint ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_epoch_loss,
        }
        torch.save(checkpoint, latest_checkpoint_path)
        print(f"✅ Checkpoint saved to {latest_checkpoint_path}")

    print("\nTraining complete!")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Dynamically load tokenizer properties ---
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Please run the tokenizer training script first.")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    actual_vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"Tokenizer loaded. Vocab size: {actual_vocab_size}, PAD token ID: {pad_id}")

    # 1. Instantiate the dataset and dataloader
    dataset = SummarizationDataset(file_path=TOKENIZED_DATA_PATH)
    # Use a lambda to pass the pad_id to the collate function
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_batch(batch, pad_id))

    # 2. Instantiate the Model with the correct vocab size
    model = SummarizationTransformer(
        vocab_size=actual_vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD
    )
    print("\nModel instantiated successfully.")

    # 3. Start the training process
    train_model(
        model, 
        dataloader, 
        vocab_size=actual_vocab_size,
        pad_id=pad_id,
        epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        save_path=MODEL_SAVE_PATH
    )

