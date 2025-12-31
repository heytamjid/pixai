"""
PIXAI model architecture for meme classification.
Adapted from PIXAI (1).ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import clip


class MultiheadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        output, _ = self.attention(query, key, value, attn_mask=mask)
        return output


class PIXAI(nn.Module):
    """
    PIXAI: Political meme classifier using vision and language features.

    Combines CLIP for visual features and XGLM for text features,
    with multi-head attention for fusion.
    """

    def __init__(self, clip_model, num_classes, num_heads, max_len):
        super(PIXAI, self).__init__()

        self.max_len = max_len  # dynamic sequence length

        # Visual feature extractor (CLIP)
        self.clip = clip_model
        self.visual_linear = nn.Linear(512, 1024)

        # Textual feature extractor (BERT/XGLM)
        self.bert = AutoModel.from_pretrained("facebook/xglm-564M")

        # Multihead attention
        self.attention = MultiheadAttention(d_model=1024, nhead=num_heads)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1024 + 1024 + 1024 + 1024, 32),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(32, num_classes),
        )

    def forward(self, image_input, input_ids, attention_mask):
        # Extract visual features
        image_features = self.clip(image_input)
        image_features = self.visual_linear(image_features)
        image_features = image_features.unsqueeze(1)

        # Dynamic pooling based on max_len
        image_features = F.adaptive_avg_pool1d(
            image_features.permute(0, 2, 1), self.max_len
        ).permute(0, 2, 1)

        # Extract Text features
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.last_hidden_state

        # Attention 1: Image querying Text
        attention_output1 = self.attention(
            query=image_features.permute(1, 0, 2),
            key=bert_output.permute(1, 0, 2),
            value=bert_output.permute(1, 0, 2),
        ).permute(1, 0, 2)

        # Attention 2: Image querying Image (Self/Cross hybrid)
        attention_output2 = self.attention(
            query=image_features.permute(1, 0, 2),
            key=bert_output.permute(1, 0, 2),
            value=image_features.permute(1, 0, 2),
        ).permute(1, 0, 2)

        # Fusion
        fusion_input = torch.cat(
            [attention_output1, attention_output2, image_features, bert_output], dim=2
        )

        output = self.fc(fusion_input.mean(1))
        return output


def load_model(
    model_path: str, device: str = "cuda", max_len: int = 64, num_heads: int = 8
):
    """
    Load the trained PIXAI model from a checkpoint.

    Args:
        model_path: Path to the .pth checkpoint file
        device: Device to load the model on ("cuda" or "cpu")
        max_len: Maximum sequence length for text
        num_heads: Number of attention heads

    Returns:
        Tuple of (model, tokenizer, clip_preprocess)
    """
    # Load CLIP model for visual features
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.visual.float().to(device)

    # Freeze CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    # Initialize model
    model = PIXAI(
        clip_model=clip_model,
        num_classes=1,  # Binary classification
        num_heads=num_heads,
        max_len=max_len,
    )
    model = model.to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")

    return model, tokenizer, preprocess


def predict(model, tokenizer, preprocess, image, text, device="cuda", max_len=64):
    """
    Make a prediction on a meme image and text.

    Args:
        model: Trained PIXAI model
        tokenizer: Text tokenizer
        preprocess: CLIP image preprocessor
        image: PIL Image
        text: Normalized text string
        device: Device to run inference on
        max_len: Maximum sequence length

    Returns:
        Dictionary with prediction and confidence score
    """
    model.eval()

    with torch.no_grad():
        # Process image
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Tokenize text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        outputs = model(image_tensor, input_ids, attention_mask)

        # Get prediction
        prob = torch.sigmoid(outputs).item()
        prediction = "Political" if prob > 0.5 else "NonPolitical"
        confidence = prob if prob > 0.5 else (1 - prob)

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "political_probability": float(prob),
            "non_political_probability": float(1 - prob),
        }
