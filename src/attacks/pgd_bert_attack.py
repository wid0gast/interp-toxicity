"""
PGD-BERT Attack for toxicity classification - Notebook version
----------------------------------------------------------
Simplified implementation for Jupyter notebook usage.
"""

import random
import logging
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize(text: str, tokenizer, device, max_length=512):
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
    """Simple dataset for text and labels."""
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class PGDBERTAttack:
    """PGD-based adversarial attack for BERT toxicity classifier."""
    
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer,
        mlm: AutoModelForMaskedLM = None,
        mlm_tokenizer = None,
        device: torch.device = None,
        max_iters: int = 10,
        top_k_tokens: int = 5,
        mlm_top_k: int = 50,
        sim_threshold: float = 0.9,
        max_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iters = max_iters
        self.top_k_tokens = top_k_tokens
        self.mlm_top_k = mlm_top_k
        self.sim_threshold = sim_threshold
        self.max_length = max_length
        
        # Initialize MLM model for generating candidates
        if mlm is None:
            print("Loading BERT MLM model for candidate generation...")
            self.mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(self.device).eval()
            self.mlm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            self.mlm = mlm
            self.mlm_tokenizer = mlm_tokenizer
        
        # Initialize sentence transformer for similarity (optional)
        self._st_model = None

    def _loss(self, logits: torch.Tensor, label: int):
        """Compute negative cross-entropy loss for targeted attack."""
        return -F.cross_entropy(logits, logits.new_tensor([label], dtype=torch.long))

    @torch.no_grad()
    def _mlm_candidates(self, tokens: List[str], idx: int) -> List[str]:
        """Get MLM replacement candidates for token at position idx."""
        try:
            masked = tokens.copy()
            masked[idx] = self.mlm_tokenizer.mask_token
            enc = tokenize(" ".join(masked), self.mlm_tokenizer, self.device, self.max_length)
            logits = self.mlm(**enc).logits[0, enc.input_ids[0] == self.mlm_tokenizer.mask_token_id]
            probs = logits.softmax(-1)
            topk = probs.topk(self.mlm_top_k, dim=-1).indices[0].tolist()
            candidates = self.mlm_tokenizer.convert_ids_to_tokens(topk)
            return [cand for cand in candidates if not cand.startswith("##") and cand != tokens[idx]]
        except Exception as e:
            print(f"Error getting MLM candidates: {e}")
            return []

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Quick cosine similarity via sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer, util
            if self._st_model is None:
                print("Loading sentence transformer for similarity checking...")
                self._st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=self.device)
            emb = self._st_model.encode([text_a, text_b], convert_to_tensor=True)
            return float(util.pytorch_cos_sim(emb[0], emb[1]))
        except ImportError:
            print("Warning: sentence-transformers not available, skipping similarity check")
            return 1.0
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 1.0

    def attack(self, text: str, label: int, verbose: bool = False) -> Tuple[str, Dict]:
        """Generate adversarial example for input text."""
        try:
            # Initial tokenization and truncation
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_length - 2:
                tokens = tokens[:self.max_length - 2]
                text = self.tokenizer.convert_tokens_to_string(tokens)
            
            enc = tokenize(text, self.tokenizer, self.device, self.max_length)
            original_text = text
            
            if verbose:
                print(f"Original text: {text}")
                print(f"Target label: {label}")
            
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
                
                if verbose:
                    current_pred = logits.argmax().item()
                    print(f"Iteration {_iter + 1}: Current prediction = {current_pred}, Loss = {loss.item():.4f}")
                
                # Get top-k influential tokens
                grad = emb.grad[0, 1:-1]  # exclude [CLS], [SEP]
                grad_norm = grad.norm(dim=-1)
                topk_idx = torch.topk(grad_norm, min(self.top_k_tokens, grad_norm.size(0))).indices.tolist()
                
                # Try MLM substitutes
                replaced = False
                best_text = text
                best_loss = loss.item()
                
                for idx in topk_idx:
                    candidates = self._mlm_candidates(tokens, idx)
                    for cand in candidates:
                        alt_tokens = tokens.copy()
                        alt_tokens[idx] = cand
                        alt_text = self.tokenizer.convert_tokens_to_string(alt_tokens)
                        
                        # Check semantic similarity
                        if self._semantic_similarity(original_text, alt_text) < self.sim_threshold:
                            continue
                            
                        alt_enc = tokenize(alt_text, self.tokenizer, self.device, self.max_length)
                        alt_logits = self.model(**alt_enc).logits
                        alt_loss = self._loss(alt_logits, label).item()
                        
                        if alt_loss < best_loss:
                            best_loss = alt_loss
                            best_text = alt_text
                            replaced = True
                            
                            # Check if attack succeeded
                            if alt_logits.argmax().item() != label:
                                if verbose:
                                    print(f"Attack succeeded! New prediction: {alt_logits.argmax().item()}")
                                    print(f"Final text: {best_text}")
                                return best_text, {"success": True, "iters": _iter + 1, "original_text": original_text}
                
                if not replaced:
                    if verbose:
                        print("No improvements found, stopping early")
                    break
                    
                text = best_text
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > self.max_length - 2:
                    tokens = tokens[:self.max_length - 2]
                    text = self.tokenizer.convert_tokens_to_string(tokens)
                enc = tokenize(text, self.tokenizer, self.device, self.max_length)
                
            # Final check
            final_enc = tokenize(text, self.tokenizer, self.device, self.max_length)
            final_pred = self.model(**final_enc).logits.argmax().item()
            success = final_pred != label
            
            if verbose:
                print(f"Final result: {'Success' if success else 'Failed'}")
                print(f"Final prediction: {final_pred}")
                print(f"Final text: {text}")
            
            return text, {"success": success, "iters": self.max_iters, "original_text": original_text}
            
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return text, {"success": False, "error": str(e), "original_text": text}

    def _convert_to_serializable(self, obj):
        """Convert tensor objects to Python native types for JSON serialization."""
        import json
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def attack_batch(self, texts: List[str], labels: List[int], batch_size: int = 1, verbose: bool = False, 
                    output_file: str = None, log_interval: int = 100) -> List[Tuple[str, Dict]]:
        """Attack a batch of texts with optional intermediate saving."""
        results = []
        total_successes = 0
        
        for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="Attacking")):
            adv_text, meta = self.attack(text, label, verbose=verbose and i < 3)  # Only verbose for first 3
            results.append((adv_text, meta))
            total_successes += int(meta["success"])
            
            # Log progress and save intermediate results
            if (i + 1) % log_interval == 0:
                print(f"Processed {i + 1}/{len(texts)}, Success rate: {total_successes/(i+1):.2%}")
            
            # Save intermediate results at specified intervals
            if output_file and (i + 1) % log_interval == 0:
                self._save_results(results, texts, labels, output_file, is_intermediate=True)
                print(f"Saved intermediate results ({len(results)} examples) → {output_file}")
        
        # Save final results
        if output_file:
            self._save_results(results, texts, labels, output_file, is_intermediate=False)
            print(f"Saved final results ({len(results)} examples) → {output_file}")
        
        final_success_rate = total_successes / len(texts)
        print(f"Final attack success rate: {final_success_rate:.2%}")
        
        return results
    
    def _save_results(self, results: List[Tuple[str, Dict]], original_texts: List[str], 
                     original_labels: List[int], output_file: str, is_intermediate: bool = False):
        """Save attack results to JSON lines format."""
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        rows = []
        for i, (adv_text, meta) in enumerate(results):
            # Convert tensors to serializable format
            meta_serializable = self._convert_to_serializable(meta)
            
            # Create result row with all metadata
            row = {
                "id": i,
                "orig_text": original_texts[i],
                "adv_text": adv_text,
                "orig_label": int(original_labels[i]),
                "success": meta_serializable.get("success", False),
                "iters": meta_serializable.get("iters", 0),
                "error": meta_serializable.get("error", None),
                "is_intermediate": is_intermediate
            }
            rows.append(row)
        
        # Save as JSONL format (one JSON object per line)
        with open(output_file, "w", encoding="utf8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

    def evaluate_robustness(self, texts: List[str], labels: List[int], sample_size: int = None, 
                           output_file: str = None, log_interval: int = 100) -> Dict:
        """Evaluate model robustness against PGD attacks."""
        if sample_size and sample_size < len(texts):
            indices = random.sample(range(len(texts)), sample_size)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        print(f"Evaluating robustness on {len(texts)} samples...")
        results = self.attack_batch(texts, labels, output_file=output_file, log_interval=log_interval)
        
        successes = sum(1 for _, meta in results if meta["success"])
        success_rate = successes / len(results)
        
        # Calculate average iterations for successful attacks
        successful_iters = [meta["iters"] for _, meta in results if meta["success"]]
        avg_iters = sum(successful_iters) / len(successful_iters) if successful_iters else 0
        
        return {
            "total_samples": len(results),
            "successful_attacks": successes,
            "attack_success_rate": success_rate,
            "average_iterations": avg_iters,
            "results": results
        }
    
    def load_saved_results(self, input_file: str) -> List[Dict]:
        """Load previously saved attack results from JSONL file."""
        import json
        results = []
        try:
            with open(input_file, "r", encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line.strip()))
            print(f"Loaded {len(results)} results from {input_file}")
        except Exception as e:
            print(f"Error loading results from {input_file}: {e}")
        return results
    
    def analyze_saved_results(self, input_file: str) -> Dict:
        """Analyze previously saved attack results."""
        results = self.load_saved_results(input_file)
        if not results:
            return {}
        
        successes = sum(1 for r in results if r.get("success", False))
        success_rate = successes / len(results)
        
        # Group by label
        toxic_results = [r for r in results if r.get("orig_label") == 1]
        non_toxic_results = [r for r in results if r.get("orig_label") == 0]
        
        toxic_success_rate = sum(1 for r in toxic_results if r.get("success", False)) / len(toxic_results) if toxic_results else 0
        non_toxic_success_rate = sum(1 for r in non_toxic_results if r.get("success", False)) / len(non_toxic_results) if non_toxic_results else 0
        
        # Average iterations for successful attacks
        successful_iters = [r.get("iters", 0) for r in results if r.get("success", False)]
        avg_iters = sum(successful_iters) / len(successful_iters) if successful_iters else 0
        
        analysis = {
            "total_samples": len(results),
            "successful_attacks": successes,
            "attack_success_rate": success_rate,
            "toxic_samples": len(toxic_results),
            "toxic_success_rate": toxic_success_rate,
            "non_toxic_samples": len(non_toxic_results),
            "non_toxic_success_rate": non_toxic_success_rate,
            "average_iterations": avg_iters,
            "results": results
        }
        
        print(f"Analysis of {len(results)} saved results:")
        print(f"Overall success rate: {success_rate:.2%}")
        print(f"Toxic examples: {toxic_success_rate:.2%} success rate ({len(toxic_results)} samples)")
        print(f"Non-toxic examples: {non_toxic_success_rate:.2%} success rate ({len(non_toxic_results)} samples)")
        print(f"Average iterations: {avg_iters:.1f}")
        
        return analysis 