import os
import pandas as pd
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split



class HyperParams:
    # Pre-training phase
    LEARNING_RATE_PRETRAIN = 0.0005
    LEARNING_RATE_FINE_TUNE = 0.001
    BATCH_SIZE = 64
    EPOCHS_PRETRAIN = 50
    EPOCHS_FINE_TUNE = 30
    DROPOUT_RATE_EMBEDDING = 0.3
    DROPOUT_RATE_GNN = 0.5
    OPTIMIZER = 'Adam'
    
    # Model architecture
    EMBEDDING_DIM = 300  # Embedding dimension for BERT
    SELF_SUPERVISED_TASKS = ['MLM', 'NSP']
    GNN_LAYERS = 2  # Number of GNN layers
    GNN_HIDDEN_UNITS = 128  # Number of hidden units in GNN
    CONTEXT_AWARE_MECHANISM = 'BiLSTM'
    ATTENTION_HEADS = 8  # Number of attention heads
    
    # Fine-tuning phase
    FINE_TUNING_BATCH_SIZE = 32
    REGULARIZATION = 0.01  # L2 regularization


# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Function to load data from a folder
def load_data_from_folder(folder_path):
    """
    Load data from CSV files in a given folder.
    Each file should contain 'text' and 'label' columns.
    """
    texts = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            texts.extend(data['text'].values)
            labels.extend(data['label'].values)
    return texts, labels

# Load training and testing data
train_texts, train_labels = load_data_from_folder("cc_2013train")
test_texts, test_labels = load_data_from_folder("cc_2013test")

# Check the number of samples loaded
print(f"Number of training samples: {len(train_texts)}")
print(f"Number of testing samples: {len(test_texts)}")


import re

def clean_text(text):
    """
    Clean text by removing punctuation and extra spaces, and converting to lowercase.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()  # Convert to lowercase

# Clean training and testing texts
train_texts = [clean_text(text) for text in train_texts]
test_texts = [clean_text(text) for text in test_texts]

# Check the cleaned text
print(f"Cleaned first training sample: {train_texts[0]}")

def encode_texts(texts, tokenizer, max_length=256):
    """
    Encode texts using BERT tokenizer.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings

# Encode training and testing texts
train_encodings = encode_texts(train_texts, tokenizer)
test_encodings = encode_texts(test_texts, tokenizer)

# Print example of encoded data
print(f"Example of training input_ids: {train_encodings['input_ids'][0]}")
print(f"Example of training attention_mask: {train_encodings['attention_mask'][0]}")


import torch
from torch_geometric.data import Data

def build_graph(texts, encodings):
    """
    Build graph data from texts and BERT encodings.
    Each word is a node, and edges are based on adjacency relationships between words.
    """
    edge_index_list = []
    x_list = []

    for text, encoding in zip(texts, encodings['input_ids']):
        # Split text into words
        words = text.split()  # Assume each word is a separate node
        num_words = len(words)
        
        # Build edges based on adjacent words
        edge_index = []
        for i in range(num_words - 1):
            edge_index.append([i, i + 1])  # Connect adjacent words
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Use BERT encodings as node features
        word_embeddings = encoding  # Use the encoded input_ids as node features
        x_list.append(torch.tensor(word_embeddings, dtype=torch.float))

        edge_index_list.append(edge_index)

    return x_list, edge_index_list

# Build graph for training and testing data
train_graphs = build_graph(train_texts, train_encodings)
test_graphs = build_graph(test_texts, test_encodings)

# Check the graph structure
print(f"Number of nodes in training graphs: {len(train_graphs[0])}")
print(f"Number of edges in training graphs: {len(train_graphs[1])}")


# Create dataset for training and testing
train_data = list(zip(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels))
test_data = list(zip(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels))

# Convert data into PyTorch Dataset format for easier loading
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create PyTorch datasets
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# Use DataLoader for batch processing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



class ContextAwareSentimentEmbeddings(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, attention_heads, dropout_rate):
        """
        Context-Aware Sentiment Embedding Layer with attention and LSTM for contextualized sentiment embeddings.
        """
        super(ContextAwareSentimentEmbeddings, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=attention_heads, dropout=dropout_rate)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, embeddings):
        """
        Forward pass for refining word embeddings with attention and LSTM-based context modeling.
        """
        attention_output, _ = self.attention(embeddings, embeddings, embeddings)
        lstm_out, _ = self.lstm(attention_output)
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out


class GNNModel(nn.Module):
    def __init__(self, embedding_dim, gnn_hidden_dim, num_layers, dropout_rate):
        """
        Graph Neural Network (GNN) Layer to capture syntactic dependencies within text.
        """
        super(GNNModel, self).__init__()
        
        self.gnn_layers = nn.ModuleList([GCNConv(embedding_dim, gnn_hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        """
        Forward pass for propagating information through the graph.
        """
        for gnn in self.gnn_layers:
            x = F.relu(gnn(x, edge_index))
            x = self.dropout(x)
        
        return x


class SentimentClassifier(nn.Module):
    def __init__(self, gnn_output_dim, num_classes, dropout_rate):
        """
        Fully connected layer for sentiment classification based on refined GNN embeddings.
        """
        super(SentimentClassifier, self).__init__()
        
        self.fc = nn.Linear(gnn_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, gnn_embeddings):
        """
        Forward pass for sentiment classification using the refined GNN embeddings.
        """
        x = self.dropout(gnn_embeddings)
        x = self.fc(x)
        return F.softmax(x, dim=-1)


class CE_GNN_Model(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, attention_heads, gnn_hidden_dim, num_layers, num_classes, dropout_rate):
        """
        Complete model integrating context-aware sentiment embeddings, GNN-based syntactic structure modeling, 
        and sentiment classification.
        """
        super(CE_GNN_Model, self).__init__()
        
        self.context_embedding = ContextAwareSentimentEmbeddings(embedding_dim, lstm_hidden_dim, attention_heads, dropout_rate)
        self.gnn = GNNModel(embedding_dim, gnn_hidden_dim, num_layers, dropout_rate)
        self.classifier = SentimentClassifier(gnn_hidden_dim, num_classes, dropout_rate)
    
    def forward(self, text_embeddings, edge_index):
        """
        Complete forward pass including context-aware sentiment embedding, GNN modeling, and classification.
        """
        context_embeddings = self.context_embedding(text_embeddings)
        gnn_output = self.gnn(context_embeddings, edge_index)
        sentiment_predictions = self.classifier(gnn_output)
        
        return sentiment_predictions


def compute_pretrain_losses(texts, tokenizer, model_mlm, model_nsp, max_length=256):
    """
    Compute MLM and NSP losses for the pre-training phase of the model.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    # MLM: Mask a percentage of words and predict
    input_ids = encodings['input_ids']
    labels = input_ids.clone()
    rand_mask = torch.rand(input_ids.shape).lt(0.15)  # 15% masking
    input_ids[rand_mask] = tokenizer.mask_token_id

    # MLM Loss
    outputs = model_mlm(input_ids, labels=labels)
    loss_mlm = outputs.loss

    # NSP: Predict if two sentences follow each other
    nsp_input_ids = tokenizer("This is a sentence.", "This is the next sentence.", return_tensors='pt', padding=True)
    nsp_labels = torch.tensor([1]).unsqueeze(0)  # Assume it's true (1: next sentence)
    outputs_nsp = model_nsp(**nsp_input_ids, labels=nsp_labels)
    loss_nsp = outputs_nsp.loss

    return loss_mlm, loss_nsp


def compute_classification_loss(predictions, labels):
    """
    Compute classification loss using cross-entropy for sentiment classification.
    """
    return F.cross_entropy(predictions, labels)


def compute_total_loss(texts, labels, tokenizer, model_mlm, model_nsp, model, edge_index, max_length=256):
    """
    Compute the total loss, combining pre-training (MLM & NSP) and sentiment classification loss.
    """
    # Compute MLM and NSP losses
    loss_mlm, loss_nsp = compute_pretrain_losses(texts, tokenizer, model_mlm, model_nsp, max_length)

    # Forward pass through the full model
    text_encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    embeddings = model.context_embedding(text_encodings['input_ids'])
    gnn_output = model.gnn(embeddings, edge_index)
    sentiment_predictions = model.classifier(gnn_output)

    # Compute classification loss
    classification_loss = compute_classification_loss(sentiment_predictions, labels)

    # Combine all losses
    total_loss = loss_mlm + loss_nsp + classification_loss
    return total_loss, loss_mlm, loss_nsp, classification_loss


def train_model(model, tokenizer, model_mlm, model_nsp, train_texts, train_labels, edge_index, num_epochs=10, learning_rate=1e-5):
    """
    Train the full model on the labeled data, including MLM and NSP pre-training and sentiment classification fine-tuning.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        total_loss, loss_mlm, loss_nsp, classification_loss = compute_total_loss(
            train_texts, train_labels, tokenizer, model_mlm, model_nsp, model, edge_index
        )
        
        # Backpropagate and update model parameters
        total_loss.backward()
        optimizer.step()
        
        # Optionally print loss details
        # print(f"Epoch {epoch+1}, Total Loss: {total_loss.item()}, MLM Loss: {loss_mlm.item()}, NSP Loss: {loss_nsp.item()}, Classification Loss: {classification_loss.item()}")


def evaluate_model(model, tokenizer, test_texts, test_labels, edge_index):
    """
    Evaluate the trained model on the test dataset using accuracy as the evaluation metric.
    """
    model.eval()
    with torch.no_grad():
        text_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
        embeddings = model.context_embedding(text_encodings['input_ids'])
        gnn_output = model.gnn(embeddings, edge_index)
        sentiment_predictions = model.classifier(gnn_output)
        
        # Get the predicted classes (highest probability)
        _, predicted_classes = torch.max(sentiment_predictions, 1)
        
        # Compute accuracy
        correct = (predicted_classes == test_labels).sum().item()
        accuracy = correct / len(test_labels)
        
    return accuracy

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model_mlm = BertForMaskedLM.from_pretrained('bert-base-chinese')
model_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')

model = CE_GNN_Model(
    embedding_dim=768, lstm_hidden_dim=128, attention_heads=8, gnn_hidden_dim=128, 
    num_layers=2, num_classes=3, dropout_rate=0.3
)


if __name__ == "__main__":
    # Hyperparameters
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model_mlm = BertForMaskedLM.from_pretrained('bert-base-chinese')
    model_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')

    model = CE_GNN_Model(
        embedding_dim=HyperParams.EMBEDDING_DIM, lstm_hidden_dim=128, 
        attention_heads=HyperParams.ATTENTION_HEADS, gnn_hidden_dim=HyperParams.GNN_HIDDEN_UNITS, 
        num_layers=HyperParams.GNN_LAYERS, num_classes=HyperParams.NUM_CLASSES, dropout_rate=HyperParams.DROPOUT_RATE
    )

    # Assuming `train_texts`, `train_labels`, `test_texts`, `test_labels`, and `edge_index` are pre-loaded datasets
    train_texts = ["我喜欢这部电影", "这是一部很有趣的电影"]  # Example texts
    train_labels = torch.tensor([0, 1])  # Example sentiment labels
    edge_index = torch.randint(0, 256, (2, 1024))  # Example edge indices for the graph

    # Train the model
    train_model(model, tokenizer, model_mlm, model_nsp, train_texts, train_labels, edge_index, num_epochs=HyperParams.EPOCHS_PRETRAIN)

    # Evaluate the model
    accuracy = evaluate_model(model, tokenizer, test_texts, test_labels, edge_index)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
