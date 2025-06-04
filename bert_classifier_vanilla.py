import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
from tqdm.auto import tqdm
import os

# Load dataset
dataset = load_dataset("csv", data_files={"train": 'jigsaw/train.csv', "test": 'jigsaw/test.csv'})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_data(example):
    return tokenizer(example["comment_text"], padding="max_length", truncation=True, max_length=128)

# Tokenize dataset
dataset = dataset.map(tokenize_data, batched=True)
dataset = dataset.rename_column("toxic", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Custom Dataset wrapper
class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: self.dataset[idx][key] for key in ["input_ids", "attention_mask", "labels"]}

# Dataloaders
batch_size = 16
train_dataloader = DataLoader(JigsawDataset(dataset["train"]), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(JigsawDataset(dataset["test"]), batch_size=batch_size, shuffle=False)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Evaluation and Early Stopping
best_accuracy = 0
patience = 3
no_improvement = 0
os.makedirs('bert_classifier_vanilla', exist_ok=True)

progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    # Training code here (unchanged)
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    print("Evaluating...")
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    accuracy = correct / total
    print(f"Epoch {epoch+1} Validation Accuracy: {accuracy:.4f}")
    
    # Save model if it has better accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        no_improvement = 0
        model_path = f'bert_classifier_vanilla/model_epoch_{epoch+1}_acc_{accuracy:.4f}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        no_improvement += 1
        
    # Early stopping
    if no_improvement >= patience:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break
