import torch
import torch.nn as nn


class SymptomClassifier(nn.Module):
    """Tiny embedding-based classifier for AfyaPlus symptom severity prediction."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, num_classes: int = 3):
        """Create the embedding layer and the final classifier."""
        super().__init__()

        # Embedding maps each token ID to a dense vector of learned numbers.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Dropout reduces overfitting on a small medical dataset.
        self.dropout = nn.Dropout(0.1)

        # One output per class: normal, urgent, emergency.
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids):
        """Encode token IDs and predict the symptom severity class."""
        # input_ids shape: [batch_size, sequence_length]
        embedded = self.embedding(input_ids)

        # Average the token vectors to get one representation per example.
        pooled = embedded.mean(dim=1)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    model = SymptomClassifier(vocab_size=100, embedding_dim=16, num_classes=3)
    sample_input = torch.tensor([[5, 12, 8], [4, 7, 9]], dtype=torch.long)
    output = model(sample_input)

    print("Model output shape:", output.shape)
    print(output)
