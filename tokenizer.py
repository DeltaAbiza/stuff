import os
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, Sequence
from tokenizers.processors import TemplateProcessing
corpus_path = "data/corpus.txt"
if not os.path.exists(corpus_path):
    print(f"Error: Corpus file not found at '{corpus_path}'")
    print("Please run the 'jsonl_corpus_extractor.py' script first to create it.")
    exit()
files = [corpus_path]
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([NFD(), Lowercase()])
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"]
)
print(f"Training the tokenizer from file: '{corpus_path}'...")
tokenizer.train(files, trainer=trainer)
print("Training complete!")
bos_token_id = tokenizer.token_to_id("[BOS]")
eos_token_id = tokenizer.token_to_id("[EOS]")

tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[
        ("[BOS]", bos_token_id),
        ("[EOS]", eos_token_id),
    ],
)
print("\n--- Testing the Tokenizer ---")
test_document = "this is a custom document we might want to summarize."
output = tokenizer.encode(test_document)

print(f"\nOriginal document: '{test_document}'")
print(f"Tokens (with BOS/EOS): {output.tokens}")
print(f"Token IDs (with BOS/EOS): {output.ids}")
print("\n--- Learned Vocabulary ---")
vocab = tokenizer.get_vocab()
for token, token_id in list(vocab.items())[:20]:
    print(f"'{token}': {token_id}")
tokenizer_path = "data/tokenizer.json"
tokenizer.save(tokenizer_path)
print(f"\nTokenizer saved to {tokenizer_path}")
loaded_tokenizer = Tokenizer.from_file(tokenizer_path)
print("\nTokenizer loaded successfully and is ready to use!")
