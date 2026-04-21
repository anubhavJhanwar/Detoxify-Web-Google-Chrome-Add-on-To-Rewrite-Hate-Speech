"""
train_model.py
--------------
Standalone script to train and save the toxicity model.
Run this once before starting the API server.

Usage:
  python train_model.py                          # uses demo data
  python train_model.py ../data/jigsaw_dataset.csv  # uses Jigsaw dataset
  python train_model.py ../data/jigsaw_dataset.csv --sample 20000 --C 1.0
"""

import sys
import argparse
from model import train_model

def main():
    parser = argparse.ArgumentParser(description="Train ToxiClear toxicity classifier")
    parser.add_argument("csv", nargs="?", default=None, help="Path to Jigsaw CSV")
    parser.add_argument("--sample", type=int, default=20000, help="Max training samples")
    parser.add_argument("--features", type=int, default=10000, help="Max TF-IDF features")
    parser.add_argument("--C", type=float, default=1.0, help="LR regularization (higher=less reg)")
    args = parser.parse_args()

    print("=" * 60)
    print("ToxiClear — Model Training")
    print("=" * 60)
    print(f"Dataset    : {args.csv or 'demo data'}")
    print(f"Sample size: {args.sample}")
    print(f"TF-IDF feat: {args.features}")
    print(f"C (reg)    : {args.C}")
    print("=" * 60)

    clf = train_model(
        jigsaw_csv=args.csv,
        sample_size=args.sample,
        max_tfidf_features=args.features,
        C=args.C,
    )

    print("\n✓ Model trained and saved to backend/models/")
    print("  You can now start the API: uvicorn app:app --reload")

if __name__ == "__main__":
    main()
