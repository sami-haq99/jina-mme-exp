import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import base64
from io import BytesIO
import re
import logging
import requests
import matplotlib.pyplot as plt
import math
from transformers import AutoTokenizer

# Standard Qwen tokenizer is compatible with Jina v4 and lightweight (CPU only)
TOKENIZER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
API_KEY = "jina_981e851b2dee47ba834256269776c26dF2quAX8oYUwN-8M7_jQapBtD-9As"
class JinaEmbeddingsClient:
    """
    Wrapper for https://api.jina.ai/v1/embeddings
    """
    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v4",
        task: str = "retrieval.query",
        timeout: int = 30,
    ) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.base_payload = {
            "model": model,
            "return_multivector": True, # Always True for grounding
            "task": task,
        }
        self.timeout = timeout

    def encode_text(self, texts: List[str]) -> List[torch.Tensor]:
        """Returns a list of tensors, one per text."""
        payload = [{"text": t} for t in texts]
        res = self._post(payload)
        return self._as_tensors(res["data"])

    def encode_image(self, images: List[Union[str, bytes, 'Image.Image']]) -> List[torch.Tensor]:
        """Returns a list of tensors, one per image."""
        def pil_image_to_base64_str(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        processed = []
        for img in images:
            if isinstance(img, bytes):
                img_str = base64.b64encode(img).decode()
            elif hasattr(img, "save"):  # PIL Image
                img_str = pil_image_to_base64_str(img)
            else:
                # Assume str URL or base64 string
                img_str = img
            processed.append({"image": img_str})

        res = self._post(processed)
        return self._as_tensors(res["data"])

    def _post(self, input_batch: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {**self.base_payload, "input": input_batch}
        try:
            resp = requests.post(
                self.API_URL, headers=self.headers, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"API Response: {e.response.text}")
            raise

    @staticmethod
    def _as_tensors(data: List[Dict[str, Any]]) -> List[torch.Tensor]:
        tensors = []
        for item in data:
            emb_lists = item["embeddings"] 
            tensors.append(torch.tensor(emb_lists, dtype=torch.float32))
        return tensors


class JinaV4SimilarityMapper:
    """
    Generates interactive similarity maps using Jina Embedding v4 API.
    Runs entirely on CPU + API (No VRAM required).
    """
    def __init__(
        self,
        api_key: str = API_KEY,
        heatmap_alpha: float = 0.6,
        device: str = "cpu" 
    ):
        self.logger = logging.getLogger("JinaV4SimMapper")
        self.device = device
        
        # 1. Initialize API Client
        self.client = JinaEmbeddingsClient(api_key=api_key)
        
        # 2. Initialize Lightweight Tokenizer (CPU Only)
        # We use the base Qwen tokenizer which matches Jina v4 but doesn't load model weights
        self.logger.info(f"Loading tokenizer: {TOKENIZER_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
        
        self.heatmap_alpha = heatmap_alpha
        self.colormap = plt.cm.get_cmap("jet")

    def process_query(self, query: str) -> Tuple[List[str], torch.Tensor, Dict[int, str]]:
        """
        1. Get vectors from API.
        2. Tokenize locally.
        3. Align vectors to tokens.
        """
        # A. API Call for Embeddings
        # Shape: [num_vectors, 128]
        # Jina API usually returns vectors for the full sequence including special tokens
        query_embeddings = self.client.encode_text([query])[0]
        
        # B. Local Tokenization
        # We need to manually add the prompt template if we want 100% alignment, 
        # but for grounding, raw tokenization is usually sufficient as the API handles the prompt.
        input_ids = self.tokenizer.encode(query)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # C. Alignment / Cleaning
        # Qwen tokenizer often uses bytes like <0x20> for spaces. We clean them.
        clean_tokens = []
        for t in tokens:
            # Replace special chars (byte fallback or Qwen specific)
            t_clean = t.replace('Ġ', '').replace(' ', '')
            try:
                # Try decoding bytes if it looks like <0x..>
                if t_clean.startswith('<0x') and t_clean.endswith('>'):
                    byte_val = int(t_clean[3:-1], 16)
                    t_clean = chr(byte_val)
            except:
                pass
            clean_tokens.append(t_clean)

        # Note: The API might return more/fewer vectors than tokens if it adds 
        # [CLS]/[EOS] internally. We assume standard behavior and truncate/pad if needed.
        # Usually Jina v4 API returns: [Vector_1, Vector_2 ... Vector_N]
        # We map them 1-to-1.
        
        num_vecs = query_embeddings.shape[0]
        num_toks = len(clean_tokens)
        
        print(f"DEBUG: Vectors received: {num_vecs}, Tokens generated: {num_toks}")

        # Heuristic alignment: 
        # If mismatch, we trust the API vector count and try to align the text tokens to it.
        # Often Jina v4 has a few prefix vectors (Task tokens) we might want to skip.
        # But for simplicity, we map directly or trim.
        
        valid_tokens = clean_tokens
        valid_embeddings = query_embeddings

        # If vectors > tokens, the API likely added prefix/suffix prompts.
        # We take the *last* N vectors matching our tokens (usually the prompt is at start).
        if num_vecs > num_toks:
            diff = num_vecs - num_toks
            # Assuming prompt vectors are at the front
            valid_embeddings = query_embeddings[diff:] 
        elif num_toks > num_vecs:
            # Rare, but truncate tokens
            valid_tokens = clean_tokens[:num_vecs]

        token_map = {i: tok for i, tok in enumerate(valid_tokens)}
        return valid_tokens, valid_embeddings, token_map

    def process_image(self, image: Union[str, bytes, Image.Image]) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        1. Get vectors from API.
        2. Infer grid dimensions (H, W) from aspect ratio and vector count.
        """
        pil_image = self._load_image(image)
        w, h = pil_image.size
        aspect_ratio = w / h

        # A. API Call
        # Shape: [num_patches, 128]
        image_embeddings = self.client.encode_image([pil_image])[0]
        num_patches = image_embeddings.shape[0]
        
        # B. Infer Grid Size (Heuristic)
        # We need to find h_grid, w_grid such that h*w = num_patches 
        # AND w/h approx aspect_ratio.
        
        # Formula: w_grid = sqrt(N * aspect)
        w_grid_est = math.sqrt(num_patches * aspect_ratio)
        h_grid_est = num_patches / w_grid_est
        
        grid_width = int(round(w_grid_est))
        grid_height = int(round(h_grid_est))
        
        # Correction: Ensure product equals exactly num_patches
        if grid_width * grid_height != num_patches:
            # Fallback 1: Try adjusting width
            if num_patches % grid_width == 0:
                grid_height = num_patches // grid_width
            # Fallback 2: Try adjusting height
            elif num_patches % grid_height == 0:
                grid_width = num_patches // grid_height
            else:
                # Fallback 3: Prime number or odd shape? 
                # Just force it to be a square-ish thing and truncate/pad later 
                # (handled in generate_heatmap)
                pass

        print(f"DEBUG: Image size {w}x{h}. Vectors: {num_patches}. Inferred Grid: {grid_width}x{grid_height}")

        return pil_image, image_embeddings, (w, h), (grid_height, grid_width)

    def _load_image(self, image: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, str):
            if image.startswith(("http://", "https://")):
                response = requests.get(image, stream=True)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content))
            else:
                pil_image = Image.open(image)
        elif isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return pil_image.convert("RGB")

    def compute_similarity_map(
        self,
        token_embedding: torch.Tensor,
        patch_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Standard Cosine Similarity
        # token: [128] -> [1, 128]
        # patches: [N, 128]
        token_norm = torch.nn.functional.normalize(token_embedding.unsqueeze(0), p=2, dim=1)
        patch_norm = torch.nn.functional.normalize(patch_embeddings, p=2, dim=1)
        
        # [1, 128] @ [128, N] -> [1, N]
        sim = torch.matmul(token_norm, patch_norm.T).squeeze(0)
        return sim

    def generate_heatmap(self, image: Image.Image, similarity_map: torch.Tensor, size: Tuple[int, int], grid_size: Tuple[int, int]) -> str:
        grid_h, grid_w = grid_size
        num_patches = similarity_map.shape[0]
        
        # Handle shape mismatch (if heuristic failed)
        if grid_h * grid_w != num_patches:
            # Force square
            side = int(math.sqrt(num_patches))
            similarity_map = similarity_map[:side*side]
            grid_h, grid_w = side, side
            
        # Normalize
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)
        
        # Reshape
        similarity_2d = similarity_map.reshape(grid_h, grid_w).cpu().numpy()
        
        # Colorize
        heatmap = (self.colormap(similarity_2d) * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap[..., :3], mode="RGB")
        
        # Resize to original image size
        heatmap_img = heatmap_img.resize(size, resample=Image.BICUBIC)
        
        # Blend
        blended = Image.blend(image.convert("RGBA"), heatmap_img.convert("RGBA"), alpha=self.heatmap_alpha)
        
        # Encode
        buffer = BytesIO()
        blended.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_token_similarity_maps(
        self,
        query: str,
        image: Union[str, bytes, Image.Image]
    ) -> Tuple[List[str], Dict[str, str]]:
        
        tokens_clean, query_vecs, token_map = self.process_query(query)
        pil_image, patch_vecs, size, grid_size = self.process_image(image)
        
        heatmaps = {}
        ui_tokens = []
        
        for idx, token_str in token_map.items():
            # Basic filter for cleaner UI
            if not token_str.strip() or len(token_str) < 2:
                continue
                
            ui_tokens.append(token_str)
            
            # Compute
            sim_map = self.compute_similarity_map(query_vecs[idx], patch_vecs)
            
            # Generate Image
            heatmap_b64 = self.generate_heatmap(pil_image, sim_map, size, grid_size)
            heatmaps[token_str] = heatmap_b64
            
        return ui_tokens, heatmaps