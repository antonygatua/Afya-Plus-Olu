from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import Tokenizer


class SymptomDataset(Dataset):
    """Dataset for AfyaPlus symptom classification examples."""

    def __init__(self, samples: List[Tuple[str, int]], tokenizer: Tokenizer):
        """Store symptom texts, labels, and the tokenizer used to encode them."""
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        """Return the number of training examples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return one sample as text, token IDs, and label."""
        text, label = self.samples[idx]
        token_ids = self.tokenizer.encode(text)

        return {
            "text": text,
            "input_ids": token_ids,
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_symptom_batch(batch):
    """Pad a batch of symptom examples to the same length."""
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    attention_mask = []

    for ids in input_ids:
        pad_len = max_len - len(ids)
        padded = torch.nn.functional.pad(
            ids,
            (0, pad_len),
            value=0,
        )
        padded_ids.append(padded)

        mask = torch.cat(
            [
                torch.ones(len(ids), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        attention_mask.append(mask)

    return {
        "texts": texts,
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": labels,
    }


def build_symptom_examples():
    """Create a wider AfyaPlus symptom dataset with realistic severity labels."""
    examples = [
        # Label 0: normal / low-risk symptoms
        ("I have a mild fever and cough", 0),
        ("I feel dizzy but I am stable", 0),
        ("I have a sore throat and mild cough", 0),
        ("I have fever for 3 days", 0),
        ("I feel weak but I can walk", 0),
        ("I have a headache and runny nose", 0),
        ("I have stomach pain after eating", 0),
        ("My child has a low fever", 0),
        ("I have body aches and fatigue", 0),
        ("I have a mild rash on my arm", 0),
        ("I feel tired but okay", 0),
        ("I have a sore throat and hoarse voice", 0),
        ("I feel nauseated but I can drink water", 0),
        ("I have a small wound on my leg", 0),
        ("I have mild swelling in my ankle", 0),
        ("I have a dry cough for 2 days", 0),
        ("I feel slightly dizzy after standing up", 0),
        ("I have mild pain in my back", 0),
        ("I have a fever and chills", 0),
        ("My throat hurts a little", 0),

        # Label 1: urgent / concerning but not immediate emergency
        ("My chest hurts and I feel pain", 1),
        ("I have severe headache and nausea", 1),
        ("I feel shortness of breath after walking", 1),
        ("I have chest pain and sweating", 1),
        ("I have trouble breathing but I am still conscious", 1),
        ("I have vomiting for more than 24 hours", 1),
        ("I feel severe abdominal pain", 1),
        ("My head is hurting badly", 1),
        ("I have a high fever and I feel weak", 1),
        ("I have severe dizziness and I am not steady", 1),
        ("My chest is tight and I feel anxious", 1),
        ("I have difficulty breathing and chest discomfort", 1),
        ("I feel faint after standing up", 1),
        ("I have a painful rash and swelling", 1),
        ("I am having repeated vomiting", 1),
        ("I have ongoing cough and pain in my chest", 1),
        ("My throat is swollen and I am struggling to swallow", 1),
        ("I have severe weakness in my legs", 1),
        ("I have chest pressure and I am sweating", 1),
        ("I feel breathless when I lie down", 1),

        # Label 2: emergency / critical distress
        ("Help! I cannot breathe", 2),
        ("Emergency! I am vomiting and have chest pain", 2),
        ("Help!!! I feel faint and weak", 2),
        ("I have chest pain and I feel like I might collapse", 2),
        ("I am having trouble breathing and my lips are blue", 2),
        ("I cannot breathe and I am panicking", 2),
        ("I am unconscious and need help", 2),
        ("I have severe bleeding and I am very weak", 2),
        ("I have a lot of blood loss after an accident", 2),
        ("My chest hurts badly and I feel like I am dying", 2),
        ("Help!!! I cannot speak properly and I am struggling to breathe", 2),
        ("I am having severe pain in the chest and I feel dizzy", 2),
        ("I am short of breath and I cannot speak full sentences", 2),
        ("I fell and I am bleeding heavily", 2),
        ("I have severe vomiting and I cannot keep fluids down", 2),
        ("Please help, I am struggling to breathe", 2),
        ("I am in pain and I cannot stop shaking", 2),
        ("This is urgent. I have trouble breathing", 2),
        ("I feel like I am going to pass out", 2),
        ("I am having severe chest pain and sweating heavily", 2),
        ("Help!!! My baby has trouble breathing", 2),
        ("I have severe dehydration and cannot stand", 2),
        ("I am very weak and my chest is tight", 2),
        ("I have intense pain in my abdomen and I am vomiting", 2),
        ("I cannot breathe and I am scared", 2),
        ("Please call an ambulance, I am struggling to breathe", 2),
        ("I am unconscious and I need urgent help", 2),
    ]
    return examples


if __name__ == "__main__":
    tokenizer = Tokenizer(min_freq=1)
    samples = build_symptom_examples()
    texts = [text for text, _ in samples]
    labels = [label for _, label in samples]

    tokenizer.fit(texts)
    dataset = SymptomDataset(samples, tokenizer)

    sample = dataset[0]
    print("Sample text:", sample["text"])
    print("Sample label:", sample["label"].item())
    print("Sample input_ids:", sample["input_ids"])

    batch = collate_symptom_batch([dataset[0], dataset[1]])
    print("Batch input_ids shape:", batch["input_ids"].shape)
    print("Batch labels:", batch["labels"])
    print("Attention mask:", batch["attention_mask"])
