# AfyaPlus Tokenizer Module

## Overview

The AfyaPlus Tokenizer module is responsible for converting raw user symptom messages into token IDs for downstream machine learning models.

It provides the text-processing layer between user input and the symptom classification pipeline.

### Pipeline

```text
User Message
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Embedding Layer
    ↓
Classifier
    ↓
Severity Classification
```

The module is designed to provide a consistent and reusable vocabulary during both model training and inference.

---

## Responsibilities

The tokenizer is responsible for:

* Normalizing input text
* Tokenizing words and punctuation
* Preserving urgency indicators such as `!`
* Handling numeric values
* Building and maintaining a vocabulary
* Mapping tokens to numerical IDs
* Encoding messages for model consumption
* Padding sequences for batch processing
* Saving and loading tokenizer metadata

The tokenizer does **not** perform symptom diagnosis, triage, or emergency classification. Its responsibility is limited to transforming text into a structured representation for downstream models.

---

## Key Design Decisions

### Punctuation Preservation

Urgency-related punctuation is preserved during tokenization.

For example:

```text
Help!!! I cannot breathe
```

may be represented as:

```text
["help", "!", "!", "!", "i", "cannot", "breathe"]
```

This allows downstream models to learn potential patterns associated with distressed or urgent messages without hard-coding prediction logic into the tokenizer.

### Domain Vocabulary

The vocabulary is built from training data and supports medical and safety-related terms, including terms such as:

* `fever`
* `cough`
* `pain`
* `chest`
* `breathe`
* `help`
* `emergency`

Vocabulary construction is configurable through parameters such as minimum token frequency.

### Vocabulary Consistency

The tokenizer used during inference must be the same tokenizer used during model training.

Vocabulary metadata can therefore be persisted and reloaded to ensure token IDs remain consistent across environments.

---

## Project Structure

```text
AfyaPlus/
└── tokenizer/
    ├── tokenizer.py     # Tokenization and vocabulary management
    ├── dataset.py       # Dataset preparation
    ├── model.py         # Embedding-based classifier
    ├── train.py         # Training entry point
    └── README.md        # Module documentation
```

---

## Usage

### Build the Vocabulary

```python
from tokenizer import Tokenizer

corpus = [
    "I have fever and cough",
    "Help! I cannot breathe",
    "My chest hurts",
]

tokenizer = Tokenizer(min_freq=1)
tokenizer.fit(corpus)
```

### Encode a Message

```python
message = "Help!!! I have chest pain"

encoded = tokenizer.encode(message)

print(encoded)
```

The output is a sequence of token IDs that can be passed to the model's embedding layer.

---

## Training

Run the training pipeline from the AfyaPlus project root:

```bash
python3 -m tokenizer.train
```

The training process is responsible for:

1. Loading the symptom dataset
2. Building or loading the tokenizer vocabulary
3. Encoding training examples
4. Training the classifier
5. Saving model and tokenizer artifacts

---

## Persistence

The tokenizer vocabulary should be saved as part of the model training artifacts.

A deployed model must always be paired with the tokenizer version used to train it.

Recommended artifact structure:

```text
artifacts/
├── model.pt
├── tokenizer.json
└── metadata.json
```

Changes to the tokenizer implementation or vocabulary-building logic should be versioned to avoid incompatibilities with previously trained models.

---

## Scope

This module currently supports the text-processing requirements for the AfyaPlus ML pipeline:

* Tokenization
* Vocabulary generation
* Token-to-ID encoding
* Dataset preparation
* Embedding-based classification
* Model training

Future changes may introduce additional preprocessing components where required by model performance or production requirements.

---

## Future Work

Potential future enhancements include:

* Mixed Swahili-English language support
* Subword tokenization
* Improved handling of spelling variations
* Tokenizer versioning
* Vocabulary monitoring
* Input validation and preprocessing metrics

A future summarization layer may also be introduced for long-form user messages. The objective would be to preserve clinically relevant symptoms and urgency indicators while reducing the amount of text processed by downstream systems.

---

## Development Notes

The current implementation prioritizes simplicity, transparency, and reproducibility.

The module is intentionally lightweight to support rapid iteration while maintaining clear boundaries between text preprocessing and model prediction logic.
