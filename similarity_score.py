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

IMG_SIZE = 1024


class JinaEmbeddingsClient:
    """
    Minimal wrapper for https://api.jina.ai/v1/embeddings
    """

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        model: str = "jina-embeddings-v4",
        return_multivector: bool = True,
        task: str = "retrieval.query",
        timeout: int = 30,
    ) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer Not Set",
        }
        self.base_payload = {
            "model": model,
            "return_multivector": return_multivector,
            "task": task,
        }
        self.timeout = timeout

    def encode_text(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Encode a batch of texts.
        """
        payload = [{"text": t} for t in texts]
        res = self._post(payload)
        return self._as_tensors(res["data"])

    def encode_image(self, images: List[Union[str, bytes, 'Image.Image']], **kwargs) -> List:
        """
        Encode a batch of images given as
        • URLs (str) – https://…/image.png  
        • base64 strings (str) – iVBORw0…  
        • raw bytes – b'\xff\xd8…' (base64‑encoded automatically)
        • PIL Image.Image instances (converted to base64 PNG)
        """
        def pil_image_to_base64_str(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        processed = []
        for img in images:
            if isinstance(img, bytes):
                img = base64.b64encode(img).decode()
            elif hasattr(img, "save"):  # PIL Image
                img = pil_image_to_base64_str(img)
            # else assume str URL or base64 string
            processed.append({"image": img})

        res = self._post(processed)
        # Assuming _post returns {'data': [...]}, convert embeddings to tensors as needed
        return [torch.tensor(item['embeddings']) for item in res['data']]

    def _post(self, input_batch: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {**self.base_payload, "input": input_batch}
        resp = requests.post(
            self.API_URL, headers=self.headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def set_api_key(self, api_key: str) -> None:
        """
        Set the API key for authentication.
        """
        if not api_key:
            raise ValueError("API key must not be empty.")
        self.headers["Authorization"] = f"Bearer {api_key}"

    @staticmethod
    def _as_tensors(data: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """
        Convert the `"data"` array of the API response into a list
        of `torch.Tensor`s (one tensor per text / image you sent).

        Each tensor’s shape is (n_vectors, dim).  When you set
        `return_multivector=False` you’ll just get shape (1, dim).
        """
        tensors: List[torch.Tensor] = []
        for item in data:                       # 1‑to‑1 with inputs
            emb_lists = item["embeddings"]      # list‑of‑lists → (N,D)
            tensors.append(torch.tensor(emb_lists, dtype=torch.float32))
        return tensors



class JinaV4SimilarityMapper:
    """
    Generates interactive similarity maps between query tokens and images using Jina Embedding v4.
    Enables visualizing which parts of an image correspond to specific words in the query.
    """
    def __init__(
        self,
        task: str = "retrieval",
        model_name: str = "jinaai/jina-embeddings-v4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        heatmap_alpha: float = 0.6,
        num_vectors: int = 128,
        client_type: str = "local",
    ):
        """
        Initialize the mapper with Jina Embedding v4.
        
        Args:
            model_name: Model name from Hugging Face hub.
            device: Compute device (GPU recommended for performance).
            patch_size: Size of image patches for embedding.
            heatmap_alpha: Transparency for the similarity heatmap.
        """
        self.task = task
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger("JinaV4SimMapper")
        self.logger.info(f"Initializing model on {device}")
        assert client_type in ["local", "web"], "client_type must be 'local' or 'web'"
        if client_type == "local":
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            self.model.eval()
        else:
            self.model = JinaEmbeddingsClient()
        self.preprocessor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.heatmap_alpha = heatmap_alpha
        self.num_vectors = num_vectors
        self.colormap = plt.cm.get_cmap("jet")  # High-contrast colormap for UI

    def process_query(self, query: str) -> Tuple[List[str], torch.Tensor, Dict[int, str]]:
        """
        Process query to get tokens, multivector embeddings, and token-index map.
        
        Args:
            query: Input query text.
            
        Returns:
            tokens: List of query tokens.
            embeddings: Multivector embeddings [num_tokens/num_vectors, embed_dim].
            token_map: Mapping from index to token.
        """
        query_embeddings = self.model.encode_text(
            texts=[query],
            task=self.task,
            prompt_name="query",
            return_multivector=True,
            truncate_dim=self.num_vectors
        )
        # Handle list vs tensor return types depending on backend
        if isinstance(query_embeddings, list):
             if isinstance(query_embeddings[0], torch.Tensor):
                query_embeddings = query_embeddings[0]
             else:
                query_embeddings = torch.tensor(query_embeddings[0])
        
        print(f"Query embeddings shape: {query_embeddings.shape}") 
        preprocessor_results = self.preprocessor.process_texts(
            texts=[query],
            prefix="Query"
        )
        input_ids = preprocessor_results["input_ids"]
        tokens = input_ids[0].tolist()
        tokens = self.preprocessor.tokenizer.convert_ids_to_tokens(tokens)
        print(f"Tokens: {tokens}")
        tokens = tokens[2:] # remove prefix
        query_embeddings = query_embeddings[2:] # remove prefix
        num_tokens = query_embeddings.shape[0]
        
        # Alignment check
        if len(tokens) != num_tokens:
             # Fallback alignment
             min_len = min(len(tokens), num_tokens)
             tokens = tokens[:min_len]
             query_embeddings = query_embeddings[:min_len]

        tokens = [tok.replace("Ġ", "") for tok in tokens]
        token_map = {i: tok for i, tok in enumerate(tokens)}
        print(f"Token map: {token_map}")
        return tokens, query_embeddings, token_map
        
    def _find_best_grid(self, num_patches: int, aspect_ratio: float) -> Tuple[int, int]:
        """
        Finds integer factors (h, w) of num_patches closest to aspect_ratio.
        """
        best_h, best_w = 1, num_patches
        min_error = float("inf")
        for h in range(1, int(math.sqrt(num_patches)) + 1):
            if num_patches % h == 0:
                w = num_patches // h
                # Option 1: h x w
                ratio_1 = w / h
                err_1 = abs(ratio_1 - aspect_ratio)
                if err_1 < min_error:
                    min_error = err_1
                    best_h, best_w = h, w
                # Option 2: w x h (swapped)
                ratio_2 = h / w
                err_2 = abs(ratio_2 - aspect_ratio)
                if err_2 < min_error:
                    min_error = err_2
                    best_h, best_w = w, h
        return best_h, best_w

    def process_image(self, image: Union[str, bytes, Image.Image]) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Process image to get patch embeddings in multivector format.
        
        Args:
            image: Image path, URL, bytes, or PIL Image.
            
        Returns:
            pil_image: Original PIL image.
            patch_embeddings: Image patch embeddings [num_patches/num_vectors, embed_dim].
            size: Original image size (width, height).
            grid_size: Patch grid dimensions (height, width) after merge.
        """
        pil_image = self._load_image(image)
        # Processor call kept for compatibility, but we ignore its grid prediction
        # as it often mismatches the actual embedding count in v4.
        self.preprocessor.process_images(images=[pil_image])
        
        size = pil_image.size
        original_width, original_height = size
        aspect_ratio = original_width / original_height

        image_embeddings = self.model.encode_image(
            images=[pil_image],
            task=self.task,
            return_multivector=True,
            max_pixels=1024*1024,
            truncate_dim=self.num_vectors
        )
        
        if isinstance(image_embeddings, list):
             if isinstance(image_embeddings[0], torch.Tensor):
                image_embeddings = image_embeddings[0]
             else:
                image_embeddings = torch.tensor(image_embeddings[0])
        
        # Remove special tokens (Logic preserved from original)
        vision_start_position_from_start = 4
        vision_end_position_from_end = 7
        image_embeddings = image_embeddings[vision_start_position_from_start:-vision_end_position_from_end]
        
        # --- FIX: Dynamic Grid Calculation ---
        # Instead of trusting the preprocessor's grid (which caused 777 vs 720 crash),
        # we calculate the best fit grid for the actual number of embeddings we have.
        num_patches = image_embeddings.shape[0]
        grid_height, grid_width = self._find_best_grid(num_patches, aspect_ratio)
        
        print(f"DEBUG: Image Patches: {num_patches}, Grid: {grid_width}x{grid_height}")

        return pil_image, image_embeddings, size, (grid_height, grid_width)

    def _load_image(self, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """Load image from various formats (URL, path, bytes, PIL Image)."""
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, str):
            if image.startswith(("http://", "https://")):
                response = requests.get(image)
                response.raise_for_status()
                pil_image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
        
        # Resize to fixed width while preserving aspect ratio
        original_width, original_height = pil_image.size
        aspect_ratio = original_height / original_width
        new_height = int(IMG_SIZE * aspect_ratio)
        pil_image = pil_image.resize((IMG_SIZE, new_height), Image.Resampling.LANCZOS)
        return pil_image

    def compute_text_text_similarity(
        self,
        text1_embeddings: torch.Tensor,
        text2_embeddings: torch.Tensor
    ) -> float:
        """
        Computes Late Interaction (MaxSim) similarity between two texts.
        """
        # 1. Normalize both sets of vectors
        t1_norm = torch.nn.functional.normalize(text1_embeddings, p=2, dim=1)
        t2_norm = torch.nn.functional.normalize(text2_embeddings, p=2, dim=1)
        
        # 2. Compute Interaction Matrix [num_tokens_1, num_tokens_2]
        # This shows how every word in T1 relates to every word in T2
        sim_matrix = torch.matmul(t1_norm, t2_norm.T)
        
        # 3. MaxSim: For every token in T1, find the best match in T2
        # Then average those best scores.
        max_scores_1_to_2 = sim_matrix.max(dim=1).values
        score = max_scores_1_to_2.mean().item()
        
        return score
    
    def calculate_multimodal_consistency(
        self,
        source: str, # Source
        candidate: str, # Target/Candidate
        image: Union[str, bytes, Image.Image]
    ) -> Dict[str, float]:
        
        # 1. Fetch all Embeddings
        _, emb_src, _ = self.process_query(source)
        _, emb_tgt, _ = self.process_query(candidate)
        _, emb_img, _, _ = self.process_image(image)
        
        # 2. Compute The Triangle Edges (Raw Scores)
        
        # A. Text1 (Source) <-> Text2 (Target)
        score_src_tgt = self.compute_text_text_similarity(emb_src, emb_tgt)

        # B. Text2 (Target) <-> Image
        # We assume image patches are the "document" (dim 1)
        tgt_norm = torch.nn.functional.normalize(emb_tgt, p=2, dim=1)
        img_norm = torch.nn.functional.normalize(emb_img, p=2, dim=1)
        sim_matrix_tgt_img = torch.matmul(tgt_norm, img_norm.T)
        score_tgt_img = sim_matrix_tgt_img.max(dim=1).values.mean().item()

        # C. Text1 (Source) <-> Image (The Baseline/Relevance)
        src_norm = torch.nn.functional.normalize(emb_src, p=2, dim=1)
        sim_matrix_src_img = torch.matmul(src_norm, img_norm.T)
        score_src_img = sim_matrix_src_img.max(dim=1).values.mean().item()

        # 3. Apply Relevance-Weighted Fusion (Academic Standard)
        # Weight lambda acts as a gate. 
        # If Source doesn't match Image (score_t1_img is low), lambda -> 0.
        # This prevents the visual part from ruining the score on bad images.
        k = 2 # Sensitivity
        lambda_weight = max(0, score_src_img) ** k

        final_score = (score_src_tgt + (lambda_weight * score_tgt_img)) / (1 + lambda_weight)
        
        mmss = 2 * (score_src_tgt * score_tgt_img) / (score_src_tgt + score_tgt_img + 1e-9)
        return {
            "MMSS": round(mmss, 4),
            "Final_Compound_Score": round(final_score, 4),
            "Text_Fidelity (T1-T2)": round(score_src_tgt, 4),
            "Visual_Grounding (T2-Img)": round(score_tgt_img, 4),
            "Image_Relevance (T1-Img)": round(score_src_img, 4),
            "Fusion_Weight": round(lambda_weight, 4)
        }
    def compute_similarity_map(
        self,
        token_embedding: torch.Tensor,
        patch_embeddings: torch.Tensor,
        aggregation: str = "mean"
    ) -> torch.Tensor:
        """
        Compute similarity between a query token and image patches.
        
        Args:
            token_embedding: Token multivector [embed_dim].
            patch_embeddings: Image patch multivectors [num_vectors/num_patches, embed_dim].
            
        Returns:
            similarity scores [num_vectors/num_patches].
        """
        num_patches = patch_embeddings.shape[0]
        token_expanded = token_embedding.expand(num_patches, -1)
        similarity_scores = torch.cosine_similarity(token_expanded, patch_embeddings, dim=1)
        return similarity_scores

    def generate_heatmap(self, image: Image.Image, similarity_map: torch.Tensor, size: Tuple[int, int], grid_size: Tuple[int, int]) -> str:
        """
        Generate a heatmap overlay on the image and return as base64.
        
        Args:
            image: Original PIL image.
            similarity_map: Similarity scores [num_patches].
            size: Original image size (width, height).
            grid_size: Patch grid dimensions (height, width).
        """
        grid_height, grid_width = grid_size
        num_patches = similarity_map.shape[0]
        required_size = grid_height * grid_width

        # Safety: If mismatch occurs (rare with dynamic grid), truncate or pad
        if num_patches > required_size:
            similarity_map = similarity_map[:required_size]
        elif num_patches < required_size:
            pad_size = required_size - num_patches
            similarity_map = torch.cat([similarity_map, torch.full((pad_size,), similarity_map.min(), device=similarity_map.device)])

        # Normalize to [0, 1]
        similarity_map = (similarity_map - similarity_map.min()) / (
            similarity_map.max() - similarity_map.min() + 1e-8
        )
        
        # Reshape to 2D grid
        similarity_2d = similarity_map.reshape(grid_height, grid_width).cpu().numpy()
        
        # Create & resize heatmap
        heatmap = (self.colormap(similarity_2d) * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap[..., :3], mode="RGB")
        heatmap = heatmap.resize(size, resample=Image.BICUBIC)
        
        # Blend with original image
        original_rgba = image.convert("RGBA")
        heatmap_rgba = heatmap.convert("RGBA")
        blended = Image.blend(original_rgba, heatmap_rgba, alpha=self.heatmap_alpha)
        
        # Encode to base64
        buffer = BytesIO()
        blended.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_token_similarity_maps(
        self,
        query: str,
        image: Union[str, bytes, Image.Image],
        aggregation: str = "mean"
    ) -> Tuple[List[str], Dict[str, str], float]:
        """
        Main method to generate similarity maps for all query tokens.
        
        Returns:
            ui_tokens: List of tokens to display
            heatmaps: Dict mapping tokens to base64 images
            global_score: Float (0-1) representing overall Text-Image Similarity (MaxSim)
        """
        _, query_embeddings, token_map = self.process_query(query)
        pil_image, patch_embeddings, size, grid_size = self.process_image(image)
        
        # --- NEW: Calculate Global Text-Image Similarity Score (MaxSim) ---
        # 1. Normalize embeddings
        q_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        p_norm = torch.nn.functional.normalize(patch_embeddings, p=2, dim=1)
        
        # 2. Compute full similarity matrix [num_tokens, num_patches]
        full_sim_matrix = torch.matmul(q_norm, p_norm.T)
        
        # 3. MaxSim: For each token, find the best matching patch (max), then average over tokens
        global_score = full_sim_matrix.max(dim=1).values.mean().item()
        
        heatmaps = {}
        tokens_for_ui = []
        
        for idx, token in token_map.items():
            if self._should_filter_token(token):
                continue   
            tokens_for_ui.append(token)
            token_embedding = query_embeddings[idx]
            sim_map = self.compute_similarity_map(
                token_embedding, patch_embeddings, aggregation
            )
            heatmap_b64 = self.generate_heatmap(pil_image, sim_map, size, grid_size)
            heatmaps[token] = heatmap_b64
        
        return tokens_for_ui, heatmaps, global_score

    def _should_filter_token(self, token: str) -> bool:
        """Filter out irrelevant tokens (punctuation, special symbols)."""
        if token.strip() == "" or re.match(r'^\s*$|^[^\w\s]+$|^<.*>$', token):
            return True
        return False