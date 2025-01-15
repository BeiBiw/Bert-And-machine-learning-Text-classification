import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import ast
from transformers import BertTokenizer
from model import BertClassfication
class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode="min", verbose=False):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric, model):
        score = -metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        """Save the current best model."""
        if self.verbose:
            print(f"Validation metric improved. Saving model...")
        torch.save(model.state_dict(), "best_model.pth")


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


def train_bert_model(model, train_loader, val_loader, epochs, lr, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, mode="min", verbose=True)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")
        for input_ids, attention_mask, labels in train_loader_tqdm:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_true_labels, all_pred_labels = [], []

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * input_ids.size(0)

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_labels.extend(predicted.cpu().numpy())

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        val_precision = precision_score(all_true_labels, all_pred_labels, average="weighted")
        val_recall = recall_score(all_true_labels, all_pred_labels, average="weighted")
        val_f1 = f1_score(all_true_labels, all_pred_labels, average="weighted")

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4%}")
        print(f"  Val Precision: {val_precision:.4%}, Val Recall: {val_recall:.4%}, Val F1 Score: {val_f1:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break
    model.load_state_dict(torch.load("best_model.pth"))
    return model

def tttest_bert_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_true_labels, all_pred_labels = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Testing"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * input_ids.size(0)

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    test_loss /= test_total
    test_accuracy = test_correct / test_total
    precision = precision_score(all_true_labels, all_pred_labels, average="weighted")
    recall = recall_score(all_true_labels, all_pred_labels, average="weighted")
    f1 = f1_score(all_true_labels, all_pred_labels, average="weighted")

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4%}")
    print(f"Precision: {precision:.4%}, Recall: {recall:.4%}, F1 Score: {f1:.4f}")


tokenizer = BertTokenizer.from_pretrained('F:/bert-base-uncased/')
def load_data(path):
    datas = pd.read_csv(path, sep=',')
    data = datas["Text"]
    label = datas["Category"].tolist()
    data = data.apply(ast.literal_eval)
    return data, label
print("Loading data......")
batch_size = 32
n_vocab = 30002
epochs = 10

class Config:
    def __init__(self, n_vocab, embed, num_filters, filter_sizes, num_classes, dropout, embedding_pretrained=None):
        self.n_vocab = n_vocab
        self.embed = embed
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.dropout = dropout  # Dropout
        self.embedding_pretrained = embedding_pretrained

config = Config(
    n_vocab=n_vocab,
    embed=256,
    filter_sizes=[3, 4, 5],
    num_classes=7,
    dropout=0.3,
    embedding_pretrained=None,
)
train_data, train_label = load_data("../../data/new_data/train.csv")
test_data, test_label = load_data("../../data/new_data/test.csv")
val_data, val_label = load_data("../../data/new_data/val.csv")

train_dataset = BertDataset(train_data, train_label,tokenizer)
test_dataset = BertDataset(test_data, test_label,tokenizer)
val_dataset = BertDataset(val_data, val_label,tokenizer)
print("Finsh Dataset......")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = BertClassfication(config)
train_bert_model(model,train_loader,val_loader,epochs,lr=0.001,patience=5)
tttest_bert_model(model,test_loader)
