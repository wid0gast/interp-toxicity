"""
PGD-BERT Attack for toxicity classification
------------------------------------------
Implements a gradient-driven word substitution attack for BERT toxicity classifier.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize(text: str, tokenizer, device, max_length=128):
    """Tokenize text with proper truncation."""
    # First truncate the text itself to avoid tokenization issues
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:  # -2 for [CLS] and [SEP]
        tokens = tokens[:max_length - 2]
        text = tokenizer.convert_tokens_to_string(tokens)
    
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class PGDBERTAttack:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer,
        mlm: AutoModelForMaskedLM,
        mlm_tokenizer,
        device: torch.device,
        max_iters: int = 10,
        top_k_tokens: int = 5,
        mlm_top_k: int = 50,
        sim_threshold: float = 0.9,
        max_length: int = 512,
        world_size: int = 1,
        rank: int = 0,
        log_interval: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_tokenizer = mlm_tokenizer
        self.device = device
        self.max_iters = max_iters
        self.top_k_tokens = top_k_tokens
        self.mlm_top_k = mlm_top_k
        self.sim_threshold = sim_threshold
        self.max_length = max_length
        self.world_size = world_size
        self.rank = rank
        self.log_interval = log_interval

        # Setup logging
        if rank == 0:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None

        # Wrap model in DDP if using multiple GPUs
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[device])

    def log(self, message: str):
        if self.rank == 0 and self.logger:
            self.logger.info(message)

    def _loss(self, logits: torch.Tensor, label: int):
        return -F.cross_entropy(logits, logits.new_tensor([label], dtype=torch.long))

    @torch.no_grad()
    def _mlm_candidates(self, tokens: List[str], idx: int) -> List[str]:
        """Get MLM replacement candidates for token at position idx."""
        masked = tokens.copy()
        masked[idx] = self.mlm_tokenizer.mask_token
        enc = tokenize(" ".join(masked), self.mlm_tokenizer, self.device, self.max_length)
        logits = self.mlm(**enc).logits[0, enc.input_ids[0] == self.mlm_tokenizer.mask_token_id]
        probs = logits.softmax(-1)
        topk = probs.topk(self.mlm_top_k, dim=-1).indices[0].tolist()
        candidates = self.mlm_tokenizer.convert_ids_to_tokens(topk)
        return [cand for cand in candidates if not cand.startswith("##") and cand != tokens[idx]]

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Quick cosine similarity via sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            return 1.0
        if not hasattr(self, "_st_model"):
            self._st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=self.device)
        emb = self._st_model.encode([text_a, text_b], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb[0], emb[1]))

    def _convert_to_serializable(self, obj):
        """Convert tensor objects to Python native types for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def attack(self, text: str, label: int) -> Tuple[str, Dict]:
        """Generate adversarial example for input text."""
        try:
            # Initial tokenization and truncation
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_length - 2:
                tokens = tokens[:self.max_length - 2]
                text = self.tokenizer.convert_tokens_to_string(tokens)
            
            enc = tokenize(text, self.tokenizer, self.device, self.max_length)
            
            for _iter in range(self.max_iters):
                # Compute gradients
                enc_ids = enc["input_ids"]
                enc_ids.requires_grad = False
                emb = self.model.get_input_embeddings()(enc_ids)
                emb.retain_grad()
                logits = self.model(inputs_embeds=emb).logits
                loss = self._loss(logits, label)
                self.model.zero_grad()
                loss.backward()
                
                # Get top-k influential tokens
                grad = emb.grad[0, 1:-1]  # exclude [CLS], [SEP]
                grad_norm = grad.norm(dim=-1)
                topk_idx = torch.topk(grad_norm, min(self.top_k_tokens, grad_norm.size(0))).indices.tolist()
                
                # Try MLM substitutes
                replaced = False
                best_text = text
                best_loss = loss.item()
                
                for idx in topk_idx:
                    for cand in self._mlm_candidates(tokens, idx):
                        alt_tokens = tokens.copy()
                        alt_tokens[idx] = cand
                        alt_text = self.tokenizer.convert_tokens_to_string(alt_tokens)
                        
                        if self._semantic_similarity(text, alt_text) < self.sim_threshold:
                            continue
                            
                        alt_enc = tokenize(alt_text, self.tokenizer, self.device, self.max_length)
                        alt_logits = self.model(**alt_enc).logits
                        alt_loss = self._loss(alt_logits, label).item()
                        
                        if alt_loss < best_loss:
                            best_loss = alt_loss
                            best_text = alt_text
                            replaced = True
                            if alt_logits.argmax().item() != label:
                                return best_text, {"success": True, "iters": _iter + 1}
                
                if not replaced:
                    break
                    
                text = best_text
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > self.max_length - 2:
                    tokens = tokens[:self.max_length - 2]
                    text = self.tokenizer.convert_tokens_to_string(tokens)
                enc = tokenize(text, self.tokenizer, self.device, self.max_length)
                
            # Final check
            final_pred = self.model(**enc).logits.argmax().item()
            success = final_pred != label
            return text, {"success": success, "iters": self.max_iters}
            
        except Exception as e:
            self.log(f"Error processing text: {str(e)}")
            return text, {"success": False, "error": str(e)}

    def attack_batch(self, texts: List[str], labels: List[int], batch_size: int = 8, output_file: str = None) -> List[Tuple[str, Dict]]:
        """Attack a batch of texts in parallel."""
        dataset = TextDataset(texts, labels)
        sampler = DistributedSampler(dataset) if self.world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=0  # Disable multiprocessing to avoid CUDA issues
        )
        print(len(dataloader))
        results = []
        total_processed = 0
        total_successes = 0
        
        for batch_idx, (batch_texts, batch_labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_results = []
            batch_successes = 0
            
            for text, label in zip(batch_texts, batch_labels):
                try:
                    adv_text, meta = self.attack(text, label)
                    batch_results.append((adv_text, meta))
                    batch_successes += int(meta["success"])
                except Exception as e:
                    self.log(f"Error in batch processing: {str(e)}")
                    batch_results.append((text, {"success": False, "error": str(e)}))
            
            results.extend(batch_results)
            total_processed += len(batch_results)
            total_successes += batch_successes
            
            # Log and save after each batch that hits the interval
            if (batch_idx + 1) * batch_size % self.log_interval == 0:
                self.log(f"Processed {total_processed} examples, Success rate: {total_successes/total_processed:.2%}")
                
                # Save intermediate results if output file is provided
                if output_file and self.rank == 0:
                    rows = []
                    for i, (adv_text, meta) in enumerate(results):
                        meta = self._convert_to_serializable(meta)  # Convert tensors to native types
                        meta.update({
                            "id": i,
                            "orig_text": texts[i],
                            "adv_text": adv_text,
                            "orig_label": int(labels[i])  # Convert tensor to int
                        })
                        rows.append(meta)
                    
                    with open(output_file, "w", encoding="utf8") as f:
                        for row in rows:
                            json.dump(row, f, ensure_ascii=False)
                            f.write("\n")
                    self.log(f"Saved intermediate results ({len(rows)} examples) → {output_file}")
            
        return results

def main():
    parser = argparse.ArgumentParser(description="PGD-BERT Attack for toxicity classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    # Initialize distributed training
    if args.world_size > 1:
        if args.local_rank == -1:
            if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
                print("Running in single GPU mode since distributed environment not set")
                args.world_size = 1
                rank = 0
            else:
                rank = int(os.environ['RANK'])
                args.world_size = int(os.environ['WORLD_SIZE'])
        else:
            rank = args.local_rank
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            
        if args.world_size > 1:
            try:
                dist.init_process_group(backend='nccl')
                print(f"Initialized process group: rank {rank}/{args.world_size}")
            except Exception as e:
                print(f"Failed to initialize distributed training: {e}")
                print("Falling back to single GPU mode")
                args.world_size = 1
                rank = 0
    else:
        rank = 0

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    set_seed()

    # Load models
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()
    
    mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device).eval()
    mlm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    import pandas as pd
    df = pd.read_csv(args.input_file)
    texts = df["generation"].tolist()
    labels = df["prompt_label"].astype(int).tolist()

    # Initialize attacker
    attacker = PGDBERTAttack(
        model, tokenizer, mlm, mlm_tokenizer, device,
        world_size=args.world_size,
        rank=rank,
        log_interval=args.log_interval,
        max_length=args.max_length
    )

    # Run attack
    results = attacker.attack_batch(
        texts, 
        labels, 
        batch_size=args.batch_size,
        output_file=args.output_file
    )
    
    # Save final results
    if rank == 0:
        rows = []
        for i, (adv_text, meta) in enumerate(results):
            if args.max_samples and i >= args.max_samples:
                break
            meta = attacker._convert_to_serializable(meta)  # Convert tensors to native types
            meta.update({
                "id": i,
                "orig_text": texts[i],
                "adv_text": adv_text,
                "orig_label": int(labels[i])  # Convert tensor to int
            })
            rows.append(meta)
            
        successes = sum(int(meta["success"]) for meta in rows)
        asr = successes / len(rows)
        print(f"\nFinal attack-success rate: {asr:.2%}")
        
        with open(args.output_file, "w", encoding="utf8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved {len(rows)} adversarial examples → {args.output_file}\n")

    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
