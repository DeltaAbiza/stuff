import json
import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast # Reverted to torch.cuda.amp for broader compatibility
import matplotlib.pyplot as plt

# --- Configuration ---
TOKENIZED_DATA_PATH = "data/tokenize_data.jsonl"
TOKENIZER_PATH = "data/tokenizer.json"
MODEL_SAVE_PATH = "./models"
BATCH_SIZE = 4
D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
LOG_INTERVAL = 10 # Log every 10 batches
MAX_SEQ_LEN = 40000
VALIDATION_SPLIT = 0.1 # Use 10% of data for validation
TRUNCATE_ARTICLE_TO=40000
# --- Step 1: Standard Dataset Class ---
class SummarizationTensorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article_ids = item["article_ids"][:TRUNCATE_ARTICLE_TO]
        return {
            "article": torch.tensor(article_ids, dtype=torch.long),
            "abstract": torch.tensor(item["abstract_ids"], dtype=torch.long)
        }

# --- Step 2: Collate Function for Padding ---
def collate_batch(batch, pad_id):
    articles = [item['article'] for item in batch]
    abstracts = [item['abstract'] for item in batch]
    padded_articles = nn.utils.rnn.pad_sequence(articles, batch_first=True, padding_value=pad_id)
    padded_abstracts = nn.utils.rnn.pad_sequence(abstracts, batch_first=True, padding_value=pad_id)
    return {"article": padded_articles, "abstract": padded_abstracts}

# --- Step 3: Transformer Model for Summarization ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SummarizationTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_mask=None):
        src_embedded = self.embedding(src) * torch.sqrt(torch.tensor(D_MODEL, dtype=torch.float32))
        src_pos = self.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_embedded = self.embedding(tgt) * torch.sqrt(torch.tensor(D_MODEL, dtype=torch.float32))
        tgt_pos = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        output = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        return nn.Transformer.generate_square_subsequent_mask(sz).to(device)

# --- Step 4: Evaluation Function ---
def evaluate(model, dataloader, criterion, device, vocab_size, pad_id):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src_batch = batch['article'].to(device)
            tgt_batch = batch['abstract'].to(device)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            
            src_padding_mask = (src_batch == pad_id).to(device)
            tgt_padding_mask = (tgt_input == pad_id).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1), device)

            output = model(src=src_batch, tgt=tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- Step 5: Training Function ---
def train_model(model, train_dataloader, val_dataloader, vocab_size, pad_id, epochs, learning_rate, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    os.makedirs(save_path, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Corrected: GradScaler() for older PyTorch versions
    scaler = GradScaler()
    start_epoch = 0
    best_val_loss = float('inf')
    
    train_losses, val_losses = [], []

    latest_checkpoint_path = os.path.join(save_path, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch}, Best validation loss so far: {best_val_loss:.4f}")

    last_val_loss = val_losses[-1] if val_losses else float('inf')

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")

        for i, batch in enumerate(train_dataloader):
            src_batch = batch['article'].to(device)
            tgt_batch = batch['abstract'].to(device)
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            src_padding_mask = (src_batch == pad_id).to(device)
            tgt_padding_mask = (tgt_input == pad_id).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1), device)

            optimizer.zero_grad()
            # Corrected: with autocast() for older PyTorch versions
            with autocast(device_type="cuda"):
                output = model(src=src_batch, tgt=tgt_input, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

            if (i + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}, Last Val Loss: {last_val_loss:.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = evaluate(model, val_dataloader, criterion, device, vocab_size, pad_id)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        last_val_loss = avg_val_loss

        print(f"--- End of Epoch {epoch+1} ---")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Average Validation Loss: {avg_val_loss:.4f}")

        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'best_val_loss': best_val_loss, 'train_losses': train_losses, 'val_losses': val_losses}
        torch.save(checkpoint, latest_checkpoint_path)
        print(f"✅ Latest checkpoint saved to {latest_checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")

    print("\nTraining complete!")
    return train_losses, val_losses

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}.")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    actual_vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"Tokenizer loaded. Vocab size: {actual_vocab_size}, PAD token ID: {pad_id}")

    print("Loading and splitting data...")
    if not os.path.exists(TOKENIZED_DATA_PATH):
        raise FileNotFoundError(f"Tokenized data file not found at {TOKENIZED_DATA_PATH}.")
    
    with open(TOKENIZED_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    split_index = int(len(all_data) * (1 - VALIDATION_SPLIT))
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]
    print(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples.")

    train_dataset = SummarizationTensorDataset(train_data)
    val_dataset = SummarizationTensorDataset(val_data)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_batch(b, pad_id), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_batch(b, pad_id))

    model = SummarizationTransformer(vocab_size=actual_vocab_size, d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD)
    print("\nModel instantiated successfully.")

    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, vocab_size=actual_vocab_size, pad_id=pad_id, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, save_path=MODEL_SAVE_PATH)

    # --- Plotting the losses ---
    print("\nPlotting training and validation losses...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, len(train_losses) + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_SAVE_PATH, "loss_plot.png"))
    print(f"Loss plot saved to {os.path.join(MODEL_SAVE_PATH, 'loss_plot.png')}")
    plt.show()
