# =====================================================
# TCRNet: Tri-Stream Cross-Reasoning Network
# =====================================================
# This code implements TCRNet as proposed in our ICDM paper.
# The network fuses three modalities:
# 1. OCR-Text features of memes (captured via BERT)
# 2. Visual features of meme images (captured via ViT)
# 3. Reasoning features (captured via Sentence Transformer)

# Key hyperparameters (from paper):
# - Epochs: 5
# - Batch Size: 16
# - Learning Rate: 2e-5
# - Dropout: 0.3
# - Attention Heads: 8
#
# NOTE: In the research, we used BERT, ViT, and MPNet encoder,
# but feel free to experiment with other models from HuggingFace.
# =====================================================

import datetime
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, BertModel, BertTokenizer,
                          ViTModel)

# ========================
# Global Configurations
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
DROPOUT = 0.3
ATTN_HEADS = 8
RESULT_FILE = "TCRNet_Results.txt"  # Metrics logged here per epoch

# ------------------------
# Task Selection
# ------------------------
# Choose between: "dark", "target", "intensity"
TASK = "dark"

# Mapping task → dataset column + number of classes
TASK_CONFIG = {
    "dark": {"label_col": "Dark", "num_labels": 2},           # Binary
    "target": {"label_col": "Target", "num_labels": 6},       # 6-class
    "intensity": {"label_col": "Intensity", "num_labels": 3}, # 3-class
}

task_info = TASK_CONFIG[TASK]
LABEL_COL = task_info["label_col"]
NUM_LABELS = task_info["num_labels"]

print(f"Using device: {DEVICE}")
print(f"Selected Task: {TASK} ({NUM_LABELS} classes)\n")

# =====================================================
# Dataset Loader
# =====================================================
class MemeDataset(Dataset):
    """
    Custom Dataset for multimodal meme classification:
      - Loads meme image, meme text, reasoning/explanation text
      - Tokenizes text inputs for BERT and reasoning encoder
    """
    def __init__(self, df, image_dir, tokenizer_bert, tokenizer_llm, transform):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer_bert = tokenizer_bert
        self.tokenizer_llm = tokenizer_llm
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["post id"]))
        image = self.transform(Image.open(img_path).convert("RGB"))

        # Tokenize meme text (meme content)
        text_input = self.tokenizer_bert(
            str(row["Text"]), return_tensors="pt",
            padding="max_length", truncation=True, max_length=128
        )
        # Tokenize reasoning/explanation text
        reasoning_input = self.tokenizer_llm(
            str(row["Explanation1"]), return_tensors="pt",
            padding="max_length", truncation=True, max_length=128
        )

        # Dynamic label selection based on TASK
        label = torch.tensor(row[LABEL_COL], dtype=torch.long)

        return image, text_input, reasoning_input, label


# =====================================================
# TCRNet Model
# =====================================================
class TCRNet(nn.Module):
    """
    Tri-Stream Cross-Reasoning Network (TCRNet):
      - Encodes three modalities: meme text, meme image, and reasoning text
      - Projects embeddings to a common space
      - Fuses them with cross-attention layers
      - Outputs classification logits
    """
    def __init__(self, bert_model, vit_model, llm_encoder, num_labels):
        super(TCRNet, self).__init__()
        # Encoders (you can experiment with other HF models here!)
        self.bert = BertModel.from_pretrained(bert_model)
        self.vit = ViTModel.from_pretrained(vit_model)
        self.llm_encoder = AutoModel.from_pretrained(llm_encoder)

        d_model = 768  # Hidden size of all encoders

        # Projection layers for each modality
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Dropout(DROPOUT)
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Dropout(DROPOUT)
        )
        self.reason_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.Dropout(DROPOUT)
        )

        # Cross-attention layers for fusion
        self.cross_attn_text = nn.MultiheadAttention(d_model, ATTN_HEADS, batch_first=True)
        self.cross_attn_vis = nn.MultiheadAttention(d_model, ATTN_HEADS, batch_first=True)
        self.cross_attn_reason = nn.MultiheadAttention(d_model, ATTN_HEADS, batch_first=True)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, num_labels)
        )

    def forward(self, image, text_input, reason_input):
        # Extract embeddings from each encoder
        text_emb = self.text_proj(
            self.bert(**{k: v.squeeze(1) for k, v in text_input.items()}).last_hidden_state[:, 0]
        )
        vis_emb = self.vis_proj(
            self.vit(pixel_values=image).last_hidden_state[:, 0]
        )
        reason_emb = self.reason_proj(
            self.llm_encoder(**{k: v.squeeze(1) for k, v in reason_input.items()}).last_hidden_state[:, 0]
        )

        # Cross-attention fusion
        t_attn, _ = self.cross_attn_text(
            text_emb.unsqueeze(1),
            torch.stack([vis_emb, reason_emb], dim=1),
            torch.stack([vis_emb, reason_emb], dim=1)
        )
        v_attn, _ = self.cross_attn_vis(
            vis_emb.unsqueeze(1),
            torch.stack([text_emb, reason_emb], dim=1),
            torch.stack([text_emb, reason_emb], dim=1)
        )
        r_attn, _ = self.cross_attn_reason(
            reason_emb.unsqueeze(1),
            torch.stack([vis_emb, text_emb], dim=1),
            torch.stack([vis_emb, text_emb], dim=1)
        )

        # Concatenate all attended features
        fused = torch.cat([t_attn.squeeze(1), v_attn.squeeze(1), r_attn.squeeze(1)], dim=1)
        return self.fc(fused)


# =====================================================
# DataLoader Helper
# =====================================================
def get_dataloaders(train_df, test_df, image_dir, bert_model, llm_encoder):
    """
    Create DataLoader objects for training and testing.
    """
    tokenizer_bert = BertTokenizer.from_pretrained(bert_model)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_encoder)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    train_ds = MemeDataset(train_df, image_dir, tokenizer_bert, tokenizer_llm, transform)
    test_ds = MemeDataset(test_df, image_dir, tokenizer_bert, tokenizer_llm, transform)
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_ds, batch_size=BATCH_SIZE)


# =====================================================
# Training and Evaluation
# =====================================================
def train_and_evaluate(model, train_loader, test_loader):
    """
    Train the model for EPOCHS and log metrics per epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.to(DEVICE)

    # Initialize log file
    with open(RESULT_FILE, "w") as f:
        f.write(f"TCRNet Experiment - Task: {TASK} - {datetime.datetime.now()}\n\n")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        for image, text_input, reason_input, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            image, label = image.to(DEVICE), label.to(DEVICE)
            for k in text_input: text_input[k] = text_input[k].to(DEVICE)
            for k in reason_input: reason_input[k] = reason_input[k].to(DEVICE)

            optimizer.zero_grad()
            output = model(image, text_input, reason_input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

        # Evaluation
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for image, text_input, reason_input, label in tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):
                image, label = image.to(DEVICE), label.to(DEVICE)
                for k in text_input: text_input[k] = text_input[k].to(DEVICE)
                for k in reason_input: reason_input[k] = reason_input[k].to(DEVICE)

                preds = model(image, text_input, reason_input).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        with open(RESULT_FILE, "a") as f:
            f.write(f"Epoch {epoch+1}: Accuracy = {acc:.4f}, Macro-F1 = {f1:.4f}\n")

    # Final classification report
    report = classification_report(all_labels, all_preds, digits=4)
    with open(RESULT_FILE, "a") as f:
        f.write(f"\nFinal Classification Report (Epoch {EPOCHS}):\n{report}\n")


# =====================================================
# Main Script
# =====================================================
if __name__ == "__main__":
    # Update paths for your setup
    train_path = "path/to/Train-Data.xlsx"
    test_path = "path/to/Test-Data.xlsx"
    image_dir = "path/to/images"

    # Model choices from the research paper
    bert_model = "google-bert/bert-base-uncased"
    vit_model = "google/vit-base-patch16-224"
    llm_model = "sentence-transformers/all-mpnet-base-v2"

    # Load datasets and dataloaders
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    train_loader, test_loader = get_dataloaders(train_df, test_df, image_dir, bert_model, llm_model)

    # Initialize and train model
    model = TCRNet(bert_model, vit_model, llm_model, NUM_LABELS)
    train_and_evaluate(model, train_loader, test_loader)

