import numpy as np
import torch
from typing import List, Dict, Union
from tqdm import tqdm
from PIL import Image
from similarity_score import JinaV4SimilarityMapper  # Assumes your file is named similarity.py

class BatchMultimodalMapper(JinaV4SimilarityMapper):
    """
    Extension of JinaV4SimilarityMapper optimized for Local GPU Batch Processing.
    """

    def calculate_corpus_metrics(
        self, 
        sources: List[str], 
        candidates: List[str], 
        images: List[Union[str, bytes]], 
        batch_size: int = 8  # Lower batch size (e.g., 4-8) recommended for GPU due to large image embeddings
    ) -> Dict[str, float]:
        
        all_sim_src_tgt = []
        all_sim_tgt_img = []
        all_sim_src_img = []

        total = len(sources)
        print(f"Processing {total} samples on {self.device} in batches of {batch_size}...")
        
        # Disable gradient calculation for inference speed
        with torch.no_grad():
            for i in tqdm(range(0, total, batch_size)):
                # 1. Prepare Batch
                batch_src = sources[i : i + batch_size]
                batch_tgt = candidates[i : i + batch_size]
                batch_img_raw = images[i : i + batch_size]

                # 2. Batch Encode
                # We use the local model directly.
                
                # A. Text Embeddings
                # Returns list of tensors [num_tokens, D]
                vecs_src = self.model.encode_text(
                    batch_src, 
                    task=self.task, 
                    prompt_name="query", 
                    return_multivector=True
                )
                
                vecs_tgt = self.model.encode_text(
                    batch_tgt, 
                    task=self.task, 
                    prompt_name="query", 
                    return_multivector=True
                )
                
                # B. Image Embeddings (Must load & resize first to match single-sample logic)
                batch_pil_images = [self._load_image(img) for img in batch_img_raw]
                
                vecs_img = self.model.encode_image(
                    batch_pil_images, 
                    task=self.task, 
                    return_multivector=True,
                    max_pixels=1024*1024, # Ensure consistency with your config
                    truncate_dim=self.num_vectors
                )
                
                # 3. Vectorized Similarity (MaxSim)
                # Compute for triplets
                all_sim_src_tgt.extend(self._batch_maxsim(vecs_src, vecs_tgt))
                all_sim_tgt_img.extend(self._batch_maxsim(vecs_tgt, vecs_img))
                all_sim_src_img.extend(self._batch_maxsim(vecs_src, vecs_img))

        # 4. Calculate Corpus Formula (Numpy)
        S_st = np.maximum(0, np.array(all_sim_src_tgt)) # Fidelity
        S_ti = np.maximum(0, np.array(all_sim_tgt_img)) # Grounding
        S_si = np.maximum(0, np.array(all_sim_src_img)) # Relevance
        
        # Lambda Gate: lambda = max(0, S_si)^2
        k = 2
        lambdas = S_si ** k
        
        # Instance Scores
        instance_scores = (S_st + (lambdas * S_ti)) / (1 + lambdas + 1e-9)
        
        return {
            "Final Score)": float(np.mean(instance_scores)),
            "Avg_Text_Fidelity": float(np.mean(S_st)),
            "Avg_Visual_Grounding": float(np.mean(S_ti)),
            "Avg_Image_Relevance": float(np.mean(S_si)),
            "Num_Samples": total
        }

    def _batch_maxsim(self, batch_a, batch_b):
        """
        Computes MaxSim score for paired batches of Multivectors.
        batch_a, batch_b: List of Tensors/Arrays
        """
        scores = []
        for v_a, v_b in zip(batch_a, batch_b):
            # Ensure tensors are on correct device
            if isinstance(v_a, np.ndarray): v_a = torch.tensor(v_a)
            if isinstance(v_b, np.ndarray): v_b = torch.tensor(v_b)
            
            v_a = v_a.to(self.device)
            v_b = v_b.to(self.device)
            
            # Normalize
            v_a = torch.nn.functional.normalize(v_a, p=2, dim=1)
            v_b = torch.nn.functional.normalize(v_b, p=2, dim=1)
            
            # Sim Matrix: [tokens_a, tokens_b]
            sim = torch.matmul(v_a, v_b.T)
            
            # MaxSim: Mean of Max(dim=1)
            score = sim.max(dim=1).values.mean().item()
            scores.append(score)
            
        return scores

# --- Usage Example ---

def load_corpus(src_lang, tgt_lang, sys, root_dir):

    sources = []
    candidates = []
    images = []

    with open(f'{root_dir}/{sys}.{src_lang}', 'r') as f:
        sources.extend([line.strip() for line in f])
    with open(f'{root_dir}/{sys}.{tgt_lang}', 'r') as f:
        candidates.extend([line.strip() for line in f])
    with open(f'{root_dir}/{images_file}', 'r') as f:
        images.extend([line.strip() for line in f])

    # Load your corpus data here
    return sources, candidates, images

if __name__ == "__main__":
    # Initialize with local GPU
    evaluator = BatchMultimodalMapper(client_type="local", device="cuda", task="retrieval")
    
    # Example Data
    
    # Example Data
    sources = ["A cat sleeping", "A fast car"]
    candidates = ["Eine schlafende Katze", "Ein schnelles Auto"]
    images = ["cat.jpg", "cat_dog.jpg"] # Local paths work best

    results = evaluator.calculate_corpus_metrics(sources, candidates, images, batch_size=4)
    print("Results:", results)
    
    """ 
    src_lang = 'en'
    tgt_lang = 'de'
    images_file = src_lang+tgt_lang+'_images.txt'  # Example file containing image paths or URLs
    sys_name = ['aya', 'zeromt']
    
    for sys in sys_name:
        sources, candidates, images = load_corpus(src_lang, tgt_lang, sys, root_dir="./corpus_data/")

        results = evaluator.calculate_corpus_metrics(sources, candidates, images, batch_size=4)
        print("Results:", results)
        #save the result to csv file, with each record on exactly 1 line to match the lines with src and tgt files
        with open(f"./corpus_data/{sys}_results.csv", "w") as f:
            for i in range(len(sources)):
                f.write(f"{sources[i]},{candidates[i]},{images[i]},{results['Final Score']},{results['Avg_Text_Fidelity']},{results['Avg_Visual_Grounding']},{results['Avg_Image_Relevance']},{results['Num_Samples']}\n") """