import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)

# These terms help the model recognise common medical symptoms and urgent contexts.
DEFAULT_MEDICAL_TERMS = [
    "fever",
    "cough",
    "chest",
    "pain",
    "headache",
    "nausea",
    "vomit",
    "breath",
    "difficult",
    "difficulty",
    "swelling",
    "rash",
    "weakness",
    "dizzy",
    "fatigue",
    "injury",
    "bleeding",
    "wound",
    "sore",
    "throat",
]

# These terms help the model detect urgency and emergency situations.
DEFAULT_SAFETY_TERMS = [
    "help",
    "urgent",
    "emergency",
    "severe",
    "danger",
    "critical",
    "cannot",
    "breathless",
    "unconscious",
    "panic",
    "call",
    "ambulance",
    "hospital",
]

# Path to the summarizer's shared emergency-term vocabulary. The tokenizer
# treats it as an optional bonus source: if it's missing (e.g. the tokenizer
# is used standalone, outside the AfyaPlus repo layout), we silently fall
# back to the defaults above instead of failing.
DEFAULT_SHARED_VOCAB_PATH = Path(__file__).resolve().parent.parent / "summarizer" / "data" / "emergency_terms.json"

# Words that are too generic to force into a safety-critical vocabulary even
# though they appear inside curated emergency phrases (e.g. "i am dying").
_PHRASE_STOPWORDS = {
    "i", "a", "an", "am", "is", "are", "was", "the", "to", "of", "and",
    "my", "me", "in", "on", "it", "has", "been", "like", "very", "feeling",
}


def _flatten_phrases(phrases: Iterable[str]) -> List[str]:
    """Split multi-word phrases into their meaningful constituent words.

    The tokenizer only ever sees individual tokens, so a phrase like
    "shortness of breath" can't be forced into the vocabulary as one unit.
    Instead we pull out its non-stopword words ("shortness", "breath") so
    they're never mapped to <unk>, while phrase-level meaning (e.g. severity
    scoring) stays the job of a dedicated phrase matcher over raw text.
    """
    words: List[str] = []
    seen = set()

    for phrase in phrases:
        for word in re.findall(r"[a-z]+", str(phrase).lower()):
            if word in _PHRASE_STOPWORDS or len(word) < 2 or word in seen:
                continue
            seen.add(word)
            words.append(word)

    return words


def _load_shared_vocab(path: Path) -> tuple[List[str], List[str]]:
    """Load extra medical/safety words from the summarizer's emergency-term data.

    Returns two empty lists (rather than raising) if the file is missing or
    malformed, since this vocabulary is a nice-to-have merge, not a hard
    dependency of the tokenizer.
    """
    if not path.exists():
        return [], []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read shared vocabulary file: %s", path)
        return [], []

    severity = data.get("severity_keywords", {})
    safety_phrases = list(data.get("emergency_terms", [])) + list(severity.get("high", []))
    medical_phrases = list(severity.get("medium", []))

    return _flatten_phrases(medical_phrases), _flatten_phrases(safety_phrases)


def _merge_unique(*term_lists: Iterable[str]) -> List[str]:
    """Union multiple term lists while preserving first-seen order."""
    merged: List[str] = []
    seen = set()
    for terms in term_lists:
        for term in terms:
            if term not in seen:
                seen.add(term)
                merged.append(term)
    return merged


class Tokenizer:
    """A practical AfyaPlus tokenizer for symptom and safety-related text."""

    def __init__(
        self,
        lower: bool = True,
        min_freq: int = 1,
        unknown_token: str = "<unk>",
        padding_token: str = "<pad>",
        start_token: str = "<bos>",
        end_token: str = "<eos>",
        medical_terms: Optional[List[str]] = None,
        safety_terms: Optional[List[str]] = None,
        shared_vocab_path: Optional[str] = DEFAULT_SHARED_VOCAB_PATH,
    ):
        """Initialize tokenizer settings.

        Args:
            medical_terms: Symptom words to force into the vocabulary.
                Defaults to DEFAULT_MEDICAL_TERMS.
            safety_terms: Urgency/emergency words to force into the
                vocabulary. Defaults to DEFAULT_SAFETY_TERMS.
            shared_vocab_path: Path to the summarizer's emergency_terms.json.
                When present, its terms are merged in on top of medical_terms/
                safety_terms (whether those are defaults or caller-supplied)
                so the tokenizer and summarizer don't drift apart. Pass None
                to skip this merge entirely.
        """
        self.lower = lower
        self.min_freq = min_freq
        self.unknown_token = unknown_token
        self.padding_token = padding_token
        self.start_token = start_token
        self.end_token = end_token

        base_medical_terms = medical_terms if medical_terms is not None else DEFAULT_MEDICAL_TERMS
        base_safety_terms = safety_terms if safety_terms is not None else DEFAULT_SAFETY_TERMS

        shared_medical_terms: List[str] = []
        shared_safety_terms: List[str] = []
        if shared_vocab_path is not None:
            shared_medical_terms, shared_safety_terms = _load_shared_vocab(Path(shared_vocab_path))

        self.medical_terms = _merge_unique(base_medical_terms, shared_medical_terms)
        self.safety_terms = _merge_unique(base_safety_terms, shared_safety_terms)

        self.special_tokens = [padding_token, unknown_token, start_token, end_token]
        self.token_to_id = {}
        self.id_to_token = {}
        self._built = False

    def normalize_text(self, text: str) -> str:
        """Lowercase and normalize user text before tokenization."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        return text.strip().lower()

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens, keeping numbers whole and punctuation single-character."""
        cleaned = self.normalize_text(text)

        # Keep numbers like 38.5 as one token; every punctuation character
        # (including runs like '!!!' or '...') becomes its own token.
        pattern = r"\d+\.\d+|\d+|[a-z]+(?:'[a-z]+)?|[!?.,;:()\-\[\]]|[^\w\s]"
        return re.findall(pattern, cleaned)

    def fit(self, corpus: Iterable[str]):
        """Build vocabulary from symptom text and keep medical/safety terms."""
        counts = Counter()

        for sample in corpus:
            tokens = self.tokenize(sample)
            counts.update(tokens)

        # Always keep medical and safety tokens in the vocabulary.
        counts.update(self.medical_terms)
        counts.update(self.safety_terms)

        valid_tokens = [token for token, count in counts.items() if count >= self.min_freq]
        valid_tokens = sorted(valid_tokens, key=lambda token: (-counts[token], token))

        ordered_tokens = []
        ordered_tokens.extend(self.special_tokens)
        ordered_tokens.extend(self.medical_terms)
        ordered_tokens.extend(self.safety_terms)
        ordered_tokens.extend(token for token in valid_tokens if token not in ordered_tokens)

        # Map each token to a unique integer ID.
        self.token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self._built = True

    def __len__(self):
        """Return vocabulary size."""
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> torch.Tensor:
        """Convert a string into a tensor of token IDs."""
        if not self._built:
            raise ValueError("Tokenizer has not been built yet. Call fit() first.")

        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(token, self.token_to_id[self.unknown_token]) for token in tokens]

        if add_special_tokens:
            ids = [self.token_to_id[self.start_token]] + ids + [self.token_to_id[self.end_token]]

        return torch.tensor(ids, dtype=torch.long)

    def encode_batch(
        self, texts: List[str], pad_to_max: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode multiple texts, padding to the same length unless pad_to_max is False.

        When pad_to_max is False, sequences keep their own lengths and are
        returned as lists rather than stacked tensors, since texts of
        different lengths can't be stacked into a single tensor.
        """
        if not self._built:
            raise ValueError("Tokenizer has not been built yet. Call fit() first.")

        encoded = [self.encode(text) for text in texts]
        max_len = max((len(ids) for ids in encoded), default=0)

        padded = []
        attention_mask = []

        for ids in encoded:
            if pad_to_max:
                pad_len = max_len - len(ids)
                padded_ids = torch.nn.functional.pad(
                    ids,
                    (0, pad_len),
                    value=self.token_to_id[self.padding_token],
                )
                mask = torch.cat(
                    [
                        torch.ones(len(ids), dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            else:
                padded_ids = ids
                mask = torch.ones(len(ids), dtype=torch.long)

            padded.append(padded_ids)
            attention_mask.append(mask)

        if pad_to_max:
            return torch.stack(padded), torch.stack(attention_mask)

        return padded, attention_mask

    def decode(self, token_ids: List[int] | torch.Tensor) -> str:
        """Convert token IDs back to a readable string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for idx in token_ids:
            token = self.id_to_token.get(int(idx), self.unknown_token)
            if token in {self.padding_token, self.start_token, self.end_token}:
                continue
            tokens.append(token)

        return " ".join(tokens)

    def save(self, filepath: str):
        """Save the tokenizer vocabulary to JSON."""
        if not self._built:
            raise ValueError("Tokenizer has not been built yet. Call fit() first.")

        data = {
            "lower": self.lower,
            "min_freq": self.min_freq,
            "unknown_token": self.unknown_token,
            "padding_token": self.padding_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_to_id": self.token_to_id,
            "medical_terms": self.medical_terms,
            "safety_terms": self.safety_terms,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(self, filepath: str):
        """Load the tokenizer vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lower = data["lower"]
        self.min_freq = data["min_freq"]
        self.unknown_token = data["unknown_token"]
        self.padding_token = data["padding_token"]
        self.start_token = data["start_token"]
        self.end_token = data["end_token"]
        self.medical_terms = data.get("medical_terms", [])
        self.safety_terms = data.get("safety_terms", [])
        self.special_tokens = [self.padding_token, self.unknown_token, self.start_token, self.end_token]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
        self._built = True


if __name__ == "__main__":
    sample_corpus = [
        "My chest hurts and I have difficulty breathing",
        "Help! I feel severe pain and fever",
        "It started 3 days ago and I feel dizzy",
        "Emergency! I am vomiting and have chest pain",
    ]

    tokenizer = Tokenizer(min_freq=1)
    tokenizer.fit(sample_corpus)

    urgent_text = "Help!!! I have chest pain!!!"
    tokens = tokenizer.tokenize(urgent_text)
    encoded = tokenizer.encode(urgent_text)

    print("Tokens:", tokens)
    print("Encoded IDs:", encoded.tolist())

    tokenizer.save("afyaplus_tokenizer.json")
    print("Saved tokenizer to afyaplus_tokenizer.json")

