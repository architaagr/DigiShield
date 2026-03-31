import os
import torch
import logging
import pandas as pd
from collections import deque
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class BERTScorer:
    def __init__(self,
                 model_path="model_artifacts/bert_finetuned",
                 base_model="distilbert-base-multilingual-cased",
                 smoothing_window=5):

        self.model_path = model_path
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Device: {self.device}")

        # Load tokenizer from saved model if it exists, else from base
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.model = None
        self.buffer = deque(maxlen=smoothing_window)

        self.SCAM_KEYWORDS = [
            "digital arrest",
            "aadhaar",
            "money laundering",
            "transfer money",
            "stay on the call",
            "legal action",
            "police case",
            "account blocked",
            "verify otp",
            "urgent transfer",
            "suspend your account"
        ]

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------
    def train(self, csv_path, epochs=3):

        logging.info("Loading dataset...")
        df = pd.read_csv(csv_path)

        # ── Accept both 'transcript' and 'text' column names ──
        if "text" not in df.columns and "transcript" in df.columns:
            df = df.rename(columns={"transcript": "text"})
            logging.info("Renamed column 'transcript' -> 'text'")

        if "text" not in df.columns:
            raise ValueError(
                f"CSV must have a 'text' or 'transcript' column. "
                f"Found: {list(df.columns)}"
            )

        # ── Class imbalance: warn and oversample minority class ──
        counts   = df["label"].value_counts().to_dict()
        n_scam   = counts.get(1, 0)
        n_benign = counts.get(0, 0)
        ratio    = max(n_scam, n_benign) / max(min(n_scam, n_benign), 1)

        logging.info(f"Dataset: {len(df)} rows | scam(1)={n_scam} | not_scam(0)={n_benign} | ratio={ratio:.1f}x")

        if ratio > 5:
            minority_label = 0 if n_benign < n_scam else 1
            majority_count = max(n_scam, n_benign)
            minority_df    = df[df["label"] == minority_label]
            repeats        = (majority_count // len(minority_df)) + 1
            oversampled    = pd.concat([minority_df] * repeats, ignore_index=True).head(majority_count)
            df = pd.concat(
                [df[df["label"] != minority_label], oversampled],
                ignore_index=True
            ).sample(frac=1, random_state=42).reset_index(drop=True)
            logging.warning(
                f"Severe imbalance detected ({ratio:.1f}x). "
                f"Oversampled minority class. New size: {len(df)} rows. "
                f"Add more real not-scam examples for best accuracy."
            )

        # ── Convert to HuggingFace Dataset ──
        dataset = Dataset.from_pandas(df[["text", "label"]])

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=128
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2
        )

        os.makedirs(self.model_path, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            per_device_train_batch_size=8,
            num_train_epochs=epochs,
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),   # auto-enable mixed precision on GPU
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        logging.info("Starting training...")
        trainer.train()

        logging.info("Saving model...")
        trainer.save_model(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    # --------------------------------------------------
    # LOAD TRAINED MODEL
    # --------------------------------------------------
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No trained model found at '{self.model_path}'.\n"
                f"Run train_model.py first."
            )

        logging.info(f"Loading trained model from {self.model_path} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        logging.info("Model loaded successfully.")

    # --------------------------------------------------
    # RESET BUFFER
    # --------------------------------------------------
    def reset_buffer(self):
        self.buffer.clear()

    # --------------------------------------------------
    # SCORE TEXT
    # --------------------------------------------------
    def score(self, text):

        if self.model is None:
            raise RuntimeError("Model not loaded. Call scorer.load() first.")

        if not text or not text.strip():
            return None

        lower_text = text.lower()

        flagged_phrases = [
            kw for kw in self.SCAM_KEYWORDS if kw in lower_text
        ]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        raw_score = probs[0][1].item()

        self.buffer.append(raw_score)
        smoothed_score = sum(self.buffer) / len(self.buffer)

        if smoothed_score >= 0.60:
            risk_level = "HIGH"
        elif smoothed_score >= 0.50:
           risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Keyword override — if 2+ known scam phrases appear, force HIGH
        #if len(flagged_phrases) >= 2:
         #   risk_level = "HIGH"

        return {
            "raw_score":       round(raw_score, 4),
            "smoothed_score":  round(smoothed_score, 4),
            "risk_level":      risk_level,
            "flagged_phrases": flagged_phrases,
        }


# --------------------------------------------------
# SINGLETON (for Streamlit / app.py)
# --------------------------------------------------

_scorer_instance = None

def get_scorer():
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = BERTScorer()
        _scorer_instance.load()
    return _scorer_instance
