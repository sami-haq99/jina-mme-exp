import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import base64
from io import BytesIO
import re
import logging
from transformers import AutoModel, AutoProcessor
import requests
import matplotlib.pyplot as plt
import os
import json
import math
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_SIZE = 1024

class JinaV4SimilarityMapper:
    """
    Unified Mapper for Single-Sample and Corpus-Batch Evaluation.
    """

    def __init__(self, client_type: str = "local", task: str = "text-matching", device: str = "cpu"):
        self.client_type = client_type
        self.device = device
        self.task = task
        self.model = None
        self.model_name = "jinaai/jina-embeddings-v4"
        
        # We need the tokenizer locally for token mapping
        try:
            self.tokenizer = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True).tokenizer
        except:
            self.tokenizer = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).tokenizer

        if client_type == "local":
            logger.info(f"Loading {self.model_name} locally on {device}...")
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(device)
            self.model.eval()
        else:
            # API Client setup
            pass 
        
        # Initialize API wrapper if needed
        self.api_client = JinaEmbeddingsClient(model=self.model_name, task=task)
        self.heatmap_alpha = 0.6 

    # ----------------------------------------------------------------
    # 1. Single Sample Logic (Reference Implementation)
    # ----------------------------------------------------------------
    def calculate_multimodal_consistency(self, source: str, candidate: str, image: Union[str, bytes, Image.Image]) -> Dict[str, float]:
        # Ensure we use local or API consistently
        if self.client_type == "local":
            return self._calculate_local_single(source, candidate, image)
        else:
            return self._calculate_api_single(source, candidate, image)

    def _calculate_local_single(self, source, candidate, image):
        """Reference logic for local calculation."""
        # 1. Load & Resize Image
        pil_image = self._load_image(image)
        
        # 2. Encode
        with torch.no_grad():
            emb_src = self.model.encode_text([source], task=self.task, return_multivector=True)[0]
            emb_tgt = self.model.encode_text([candidate], task=self.task, return_multivector=True)[0]
            emb_img = self.model.encode_image([pil_image], task=self.task, return_multivector=True)[0]
            
            # 3. SLICING (Crucial Step)
            # Remove Jina V4 Vision Special Tokens (Start: 4, End: 7)
            emb_img = emb_img[4:-7]
            
            # 4. Calculate MaxSim
            return self._compute_mmss_metrics(emb_src, emb_tgt, emb_img)

    def _compute_mmss_metrics(self, v_src, v_tgt, v_img):
        """Shared math logic."""
        s_fidelity = self._calc_single_maxsim(v_src, v_tgt)
        s_grounding = self._calc_single_maxsim(v_tgt, v_img)
        s_relevance = self._calc_single_maxsim(v_src, v_img)

        # Formula
        k_factor = 2
        lamb = max(0, s_relevance) ** k_factor
        

        final_score = (s_fidelity + (lamb * s_grounding)) / (1 + lamb + 1e-9)
        
        return {
            "Final_Weighted_Score": round(final_score, 4),
            "Text_Fidelity": round(s_fidelity, 4),
            "Visual_Grounding": round(s_grounding, 4),
            "Image_Relevance": round(s_relevance, 4)
        }

    # ----------------------------------------------------------------
    # 2. Batch Corpus Logic (Optimized)
    # ----------------------------------------------------------------
    def calculate_individual_scores(
        self, 
        sources: List[str], 
        candidates: List[str], 
        images: List[str], 
        batch_size: int = 8,
        output_file: str = "corpus_results.csv"
    ):
        """
        Calculates scores for a list of items using GPU batching.
        Strictly aligns preprocessing with single-sample logic.
        """
        if self.client_type != "local":
            raise ValueError("Corpus processing requires client_type='local' with GPU.")

        results_data = []
        total = len(sources)
        
        print(f"🚀 Processing {total} samples on {self.device}...")
        
        with torch.no_grad():
            for i in tqdm(range(0, total, batch_size), desc="Batch Processing"):
                b_src = sources[i : i + batch_size]
                b_tgt = candidates[i : i + batch_size]
                b_img_paths = images[i : i + batch_size]
                
                # A. Encode Text
                vecs_src = self.model.encode_text(b_src, task=self.task, return_multivector=True)
                vecs_tgt = self.model.encode_text(b_tgt, task=self.task, return_multivector=True)
                
                # B. Load & Resize Images (Match Single Logic)
                b_pil_imgs = []
                for p in b_img_paths:
                    try:
                        b_pil_imgs.append(self._load_image(p)) # Use _load_image!
                    except:
                        b_pil_imgs.append(Image.new('RGB', (1024, 1024)))

                # C. Encode Images
                vecs_img_raw = self.model.encode_image(
                    b_pil_imgs, 
                    task=self.task, 
                    return_multivector=True,
                    max_pixels=1024*1024
                )

                # D. Pairwise Calc + SLICING
                for k in range(len(b_src)):
                    v_s = torch.tensor(vecs_src[k]).to(self.device)
                    v_t = torch.tensor(vecs_tgt[k]).to(self.device)
                    v_i = torch.tensor(vecs_img_raw[k]).to(self.device)
                    
                    # --- CRITICAL FIX: SLICE IMAGE TOKENS ---
                    # Match the [4:-7] logic from single sample
                    if v_i.shape[0] > 11: # Safety check
                        v_i = v_i[4:-7]
                    # ----------------------------------------

                    metrics = self._compute_mmss_metrics(v_s, v_t, v_i)
                    
                    row = {
                        "id": i + k,
                        "source": b_src[k],
                        "candidate": b_tgt[k],
                        "image": b_img_paths[k],
                        **metrics
                    }
                    results_data.append(row)

        df = pd.DataFrame(results_data)
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        return df

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _load_image(self, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """Standardized image loading and resizing."""
        if isinstance(image, str):
            if image.startswith("http"):
                pil_image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError("Unknown image type")
            
        # Resize logic matching your original script
        original_width, original_height = pil_image.size
        aspect_ratio = original_height / original_width
        new_height = int(IMG_SIZE * aspect_ratio)
        pil_image = pil_image.resize((IMG_SIZE, new_height), Image.Resampling.LANCZOS)
        return pil_image

    def _calc_single_maxsim(self, t1, t2):
        """Calculates Late Interaction (MaxSim) similarity."""
        if isinstance(t1, np.ndarray): t1 = torch.tensor(t1)
        if isinstance(t2, np.ndarray): t2 = torch.tensor(t2)
        
        t1 = t1.to(self.device)
        t2 = t2.to(self.device)

        t1 = torch.nn.functional.normalize(t1, p=2, dim=1)
        t2 = torch.nn.functional.normalize(t2, p=2, dim=1)
        
        sim_matrix = torch.matmul(t1, t2.T)
        return sim_matrix.max(dim=1).values.mean().item()

# Placeholder for API Client if needed
class JinaEmbeddingsClient:
    def __init__(self, model, task): pass 
    def encode_text(self, x): pass
    def encode_image(self, x): pass
    
    
if __name__ == "__main__":
    from PIL import Image # Ensure PIL is available
    
    # 1. Setup Data (Example)
    # In real usage, load these from a file
    sources = ["We'll have to get rid of that mole.", "He finally made it to the bank."] * 10 # 100 samples
    candidates = ["Wir müssen uns von diesem Maulwurf trennen.", "Er kam endlich an der Bank an."] * 10
    images = [ "mole.jpeg", "bank.jpeg"] * 10 # Ensure these files exist locally!

    
    mapper = JinaV4SimilarityMapper(task = 'retrieval') 
    results = mapper.calculate_individual_scores(sources, candidates, images)
    #example result values:
    
    print(results.head())

