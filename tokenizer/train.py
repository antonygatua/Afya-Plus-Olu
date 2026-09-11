import torch
from torch.utils.data import DataLoader

from tokenizer.dataset import SymptomDataset, build_symptom_examples, collate_symptom_batch
from tokenizer.model import SymptomClassifier
from tokenizer.tokenizer import Tokenizer


def main():
    """Train the symptom classifier using the AfyaPlus tokenizer and dataset."""
    samples = build_symptom_examples()
    texts = [text for text, _ in samples]

    tokenizer = Tokenizer(min_freq=1)
    tokenizer.fit(texts)

    dataset = SymptomDataset(samples, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_symptom_batch,
    )

    model = SymptomClassifier(
        vocab_size=len(tokenizer),
        embedding_dim=32,
        num_classes=3,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train for a few epochs on this small symptom dataset.
    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss = {total_loss:.4f}")

    # Evaluate on the same training data for a quick sanity check.
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["input_ids"])
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Training accuracy: {accuracy:.4f}")

    # Show a few test predictions on symptom messages.
    test_texts = [
        "I have a severe headache and feel dizzy",
        "I am experiencing shortness of breath",
        "My throat is very sore and I have difficulty swallowing",
        "I feel extremely weak and my body is aching",
        "I have a high fever and feel very weak!",
        "Help! My cough is getting worse and I feel dizzy!"
    ]

    for text in test_texts:
        input_ids = tokenizer.encode(text).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)
            prediction = torch.argmax(logits, dim=1).item()
        print(f"Text: {text} -> predicted label: {prediction}")


if __name__ == "__main__":
    main()
