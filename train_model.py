# ============================================================
# train_model.py
# Run this ONCE before starting the app.
#
# Usage:
#   python train_model.py
#
# What it does:
#   1. Loads your CSV
#   2. Fine-tunes DistilBERT on it
#   3. Saves checkpoint to model_artifacts/bert_finetuned/
# ============================================================

from nlp_scorer import BERTScorer

scorer = BERTScorer()

scorer.train(
    csv_path="your_dataset.csv",  # your CSV filename
    epochs=3,
)

print("\nTraining complete.")
print("Now run:  streamlit run app.py")
print("  or   :  python realtime_asr.py   (for terminal-only mode)")
