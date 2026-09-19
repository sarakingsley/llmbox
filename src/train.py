import logging
from torch.utils.data import DataLoader
from data_services import TrainingDataLoader



training_data = TrainingDataLoader(
    assistant_mask_strategy="template",
)

conversations = training_data._load_conversations(cfg)

dataset = training_data._build_dataset(
    conversations=conversations,
    tokenizer=tokenizer,
    max_length=2048,
)

pad_token_id = tokenizer.pad_token_id

if pad_token_id is None:
    # Reusing EOS for padding is common for decoder-only training.
    # Padding labels are ignored and padding attention positions are zero.
    pad_token_id = tokenizer.eos_token_id

if pad_token_id is None:
    raise ValueError(
        "Provide a padding token ID; this tokenizer has neither PAD nor EOS."
    )

data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=training_data._make_collator(pad_token_id),
)

batch = next(iter(data_loader))

print("Input shape:", batch["input_ids"].shape)
print("Label shape:", batch["labels"].shape)
print("Trainable targets:", (batch["labels"] != -100).sum().item())

# Hugging Face causal-LM models generally shift labels internally:
# outputs = model(**batch)
# loss = outputs.loss

# If the tokenizer’s template does not support assistant masks, explicitly try:
training_data = TrainingDataLoader(
    assistant_mask_strategy="verified_prefix",
)
