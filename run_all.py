"""
run_all.py — End-to-end execution script.

Executes:
1. Directory structure verification
2. VAE training
3. DCGAN training
4. Evaluation and comparative metrics calculation
5. Plot generation
"""

import os
import sys
import argparse

import config
from utils.helpers import ensure_dirs, set_seed
from training.train_vae import train_vae
from training.train_gan import train_gan
from evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Run full VAE + GAN pipeline")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training")
    parser.add_argument("--skip-gan", action="store_true", help="Skip GAN training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    set_seed(config.SEED)
    ensure_dirs()

    print("\n=======================================================")
    print("  GENERATIVE IMAGE SYNTHESIS (VAE & DCGAN) PIPELINE  ")
    print("=======================================================")
    print(f"Device: {config.DEVICE}")
    print(f"Dataset configured: {config.DATASET}")
    print(f"Image resolution: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}\n")

    if not args.eval_only:
        if not args.skip_vae:
            print("\n>>> STEP 1: TRAINING VAE >>>")
            train_vae()
        else:
            print("\n>>> STEP 1: Skipping VAE training")

        if not args.skip_gan:
            print("\n>>> STEP 2: TRAINING DCGAN >>>")
            train_gan()
        else:
            print("\n>>> STEP 2: Skipping GAN training")

    print("\n>>> STEP 3: EVALUATING MODELS & GENERATING COMPARISONS >>>")
    evaluate()

    print("\n=======================================================")
    print("  PIPELINE COMPLETE!  ")
    print("  Outputs saved to: outputs/")
    print("  To launch the demo UI, run: streamlit run app.py")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
