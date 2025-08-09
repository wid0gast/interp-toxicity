#!/usr/bin/env python3
"""
Model Robustness Evaluation Script
==================================
Evaluates model robustness against PGD attacks on toxicity classification.

This script converts the notebook section for evaluating model robustness
with PGD attacks into a standalone Python script.
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pgd_bert_attack import PGDBERTAttack, set_seed
from tqdm import tqdm
import argparse


def setup_device():
    """Setup CUDA device and print available devices."""
    print("Available CUDA devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    return device


def load_model_and_tokenizer():
    """Load the toxicity classification model and tokenizer."""
    print("Loading toxicity classification model...")
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta_toxicity_classifier")
    return model, tokenizer


def load_data(data_path):
    """Load the test dataset."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    return df


def evaluate_robustness(model, tokenizer, device, df, sample_size=100, output_file=None):
    """
    Evaluate model robustness against PGD attacks.
    
    Args:
        model: The toxicity classification model
        tokenizer: The tokenizer for the model
        device: CUDA device to use
        df: DataFrame containing the test data
        sample_size: Number of samples to evaluate (default: 100)
        output_file: Path to save results (default: None)
    
    Returns:
        dict: Robustness evaluation results
    """
    print("Evaluating model robustness with PGD attacks...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize PGD Attack
    print("Setting up PGD BERT Attack...")
    
    # Create the attacker - it will automatically load BERT MLM model for candidate generation
    attacker = PGDBERTAttack(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_iters=10,
        top_k_tokens=5,
        mlm_top_k=50,
        sim_threshold=0.9,  # Slightly lower threshold for more flexibility
        max_length=512
    )
    
    print("PGD Attack setup complete!")
    
    # Sample some data for evaluation
    sample_df = df.sample(sample_size, random_state=42)
    sample_texts = sample_df.comment_text.tolist()
    sample_labels = sample_df.toxicity_label.astype(int).tolist()
    
    print(f"Running attacks on {sample_size} samples...")
    print(f"Sample distribution: {sum(sample_labels)} toxic, {len(sample_labels) - sum(sample_labels)} non-toxic")
    
    # Set default output file if not provided
    if output_file is None:
        output_file = "snlp_roberta/jigsaw_pgd.jsonl"
    
    # Run batch attack evaluation with saving enabled
    robustness_results = attacker.evaluate_robustness(
        sample_texts, 
        sample_labels,
        output_file=output_file,
        log_interval=100  # Save every 100 examples
    )
    
    # Print results
    print(f"\nRobustness Evaluation Results:")
    print(f"Total samples: {robustness_results['total_samples']}")
    print(f"Successful attacks: {robustness_results['successful_attacks']}")
    print(f"Attack success rate: {robustness_results['attack_success_rate']:.2%}")
    print(f"Average iterations for successful attacks: {robustness_results['average_iterations']:.1f}")
    print(f"Results saved to: {output_file}")
    
    return robustness_results


def analyze_results(robustness_results, sample_labels):
    """Analyze and display detailed attack results."""
    print("Analysis of Attack Results:")
    print("=" * 50)
    
    results = robustness_results['results']
    successful_attacks = [(i, adv_text, meta) for i, (adv_text, meta) in enumerate(results) if meta['success']]
    
    if successful_attacks:
        print(f"\nShowing first 5 successful attacks:")
        print("-" * 40)
        
        for i, (idx, adv_text, meta) in enumerate(successful_attacks[:5]):
            original_text = meta['original_text']
            print(f"\nExample {i+1}:")
            print(f"Original: {original_text}")
            print(f"Adversarial: {adv_text}")
            print(f"Original label: {sample_labels[idx]}")
            print(f"Iterations: {meta['iters']}")
            
            # Check the difference
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, original_text, adv_text).ratio()
            print(f"Text similarity: {similarity:.3f}")
            print("-" * 40)
    else:
        print("No successful attacks found in this sample.")
    
    # Attack success by label
    print(f"\nAttack Success by Original Label:")
    toxic_attacks = [(adv_text, meta) for i, (adv_text, meta) in enumerate(results) if sample_labels[i] == 1]
    non_toxic_attacks = [(adv_text, meta) for i, (adv_text, meta) in enumerate(results) if sample_labels[i] == 0]
    
    if toxic_attacks:
        toxic_success_rate = sum(1 for _, meta in toxic_attacks if meta['success']) / len(toxic_attacks)
        print(f"Toxic examples: {toxic_success_rate:.2%} success rate ({len(toxic_attacks)} samples)")
    
    if non_toxic_attacks:
        non_toxic_success_rate = sum(1 for _, meta in non_toxic_attacks if meta['success']) / len(non_toxic_attacks)
        print(f"Non-toxic examples: {non_toxic_success_rate:.2%} success rate ({len(non_toxic_attacks)} samples)")
    
    print(f"\nOverall model robustness: {100 - robustness_results['attack_success_rate']*100:.1f}% robust to PGD attacks")


def main():
    """Main function to run the robustness evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model robustness against PGD attacks")
    parser.add_argument("--data_path", type=str, default="jigsaw/test_clean.csv", 
                       help="Path to the test data CSV file")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="Number of samples to evaluate (default: 100)")
    parser.add_argument("--output_file", type=str, default="snlp_roberta/jigsaw_dem_pgd.jsonl",
                       help="Path to save results (default: snlp_roberta/jigsaw_pgd.jsonl)")
    parser.add_argument("--device_id", type=int, default=5,
                       help="CUDA device ID to use (default: 5)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() else torch.device('cpu')
    setup_device()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    model = model.to(device)
    model.eval()
    
    # Load data
    df = load_data(args.data_path)
    
    # Sample data for evaluation
    sample_df = df.sample(args.sample_size, random_state=42)
    sample_texts = sample_df.comment_text.tolist()
    sample_labels = sample_df.toxicity_label.astype(int).tolist()
    
    # Evaluate robustness
    robustness_results = evaluate_robustness(
        model=model,
        tokenizer=tokenizer,
        device=device,
        df=df,
        sample_size=args.sample_size,
        output_file=args.output_file
    )
    
    # Analyze results
    analyze_results(robustness_results, sample_labels)


if __name__ == "__main__":
    main() 