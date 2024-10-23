import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence

# Simple tokenizer function to split text into words


def tokenizer(text):
    # Basic tokenization: split by spaces and remove non-alphabetic characters
    return re.findall(r'\b\w+\b', text.lower())

# Custom dataset class for text data


class TextDataset(Dataset):
    def __init__(self, csv_file, text_column, label_column, max_seq_len=50):
        self.data = pd.read_csv(csv_file)
        self.text_column = text_column
        self.label_column = label_column
        self.max_seq_len = max_seq_len

        # Define label mapping (adjust according to your labels)
        label_mapping = {'negative': 0, 'neutral': 1,
                         'positive': 2}  # Example mapping
        self.data[label_column] = self.data[label_column].map(label_mapping)

        # Check for any NaN values after mapping
        if self.data[label_column].isnull().any():
            raise ValueError(
                "Some labels could not be mapped. Please check your label values.")

        # Build vocabulary
        self.vocab, self.inv_vocab = self.build_vocab()
        self.texts = [self.process_text(text)
                      for text in self.data[self.text_column]]

        # Convert labels to a numeric tensor
        self.labels = torch.tensor(
            self.data[self.label_column].values, dtype=torch.float32)

    def build_vocab(self):
        counter = Counter()
        for text in self.data[self.text_column]:
            tokens = tokenizer(text)
            counter.update(tokens)

        # Create vocabulary mapping
        # Start indexing from 2
        vocab = {word: idx + 2 for idx,
                 (word, _) in enumerate(counter.items())}
        vocab['<pad>'] = 0  # Padding token
        vocab['<unk>'] = 1  # Unknown token

        inv_vocab = {idx: word for word, idx in vocab.items()}
        return vocab, inv_vocab

    def process_text(self, text):
        tokens = tokenizer(text)
        token_ids = [self.vocab.get(token, self.vocab['<unk>'])
                     for token in tokens][:self.max_seq_len]
        # Pad with <pad> token if necessary
        token_ids += [self.vocab['<pad>']] * \
            (self.max_seq_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# Collate function to handle padding within DataLoader


def collate_fn(batch):
    texts, labels = zip(*batch)
    # Padding with <pad> token (index 0)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels

# SimpleRNN model for text classification


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, padding_idx=0):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)  # out: (batch_size, seq_length, hidden_size)
        # Only use the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# Function to calculate accuracy


def calculate_accuracy(y_true, y_pred):
    y_pred = torch.round(torch.sigmoid(y_pred))
    return accuracy_score(y_true.cpu(), y_pred.cpu())

# Function to train and evaluate the model


def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device):
    train_acc_history, valid_acc_history = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Training loop
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(
                device).unsqueeze(1)  # Ensure labels match output shape
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        valid_loss, valid_correct = 0, 0
        model.eval()
        all_labels, all_outputs = [], []

        with torch.no_grad():
            for texts, labels in valid_loader:
                texts, labels = texts.to(
                    device), labels.to(device).unsqueeze(1)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                all_labels.append(labels)
                all_outputs.append(outputs)

        all_labels = torch.cat(all_labels)
        all_outputs = torch.cat(all_outputs)

        # Calculate accuracy
        train_accuracy = calculate_accuracy(all_labels, all_outputs)
        valid_accuracy = calculate_accuracy(all_labels, all_outputs)

        train_acc_history.append(train_accuracy)
        valid_acc_history.append(valid_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Train Acc: {train_accuracy:.4f}, Valid Acc: {valid_accuracy:.4f}')

    return train_acc_history, valid_acc_history

# Save accuracy metrics to a file


def save_metrics(train_acc_history, valid_acc_history, filename='accuracy_metrics.csv'):
    df = pd.DataFrame({
        'train_accuracy': train_acc_history,
        'valid_accuracy': valid_acc_history
    })
    df.to_csv(filename, index=False)


# Parameters
text_column = 'text'
label_column = 'label'
embedding_dim = 100
hidden_size = 50
output_size = 1
batch_size = 32
num_epochs = 1000
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
data_dir = "./datasets/sentiment/indonesian/"
train_dataset = TextDataset(data_dir + 'train.csv', text_column, label_column)
valid_dataset = TextDataset(
    data_dir + 'valid.csv', text_column, label_column, max_seq_len=train_dataset.max_seq_len)
test_dataset = TextDataset(data_dir + 'test.csv', text_column,
                           label_column, max_seq_len=train_dataset.max_seq_len)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize the model, loss function, and optimizer
vocab_size = len(train_dataset.vocab)
model = SimpleRNN(vocab_size, embedding_dim,
                  hidden_size, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model and save accuracy metrics
train_acc_history, valid_acc_history = train_model(
    model, criterion, optimizer, train_loader, valid_loader, num_epochs, device)
save_metrics(train_acc_history, valid_acc_history)

# Test the model


def test_model(model, test_loader, device):
    model.eval()
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            all_labels.append(labels)
            all_outputs.append(outputs)

    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    test_accuracy = calculate_accuracy(all_labels, all_outputs)
    print(f'Test Accuracy: {test_accuracy:.4f}')


test_model(model, test_loader, device)
