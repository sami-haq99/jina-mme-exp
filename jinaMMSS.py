import requests
import torch
import torch.nn.functional as F

class JinaAPIv4Scorer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def get_multivectors(self, inputs):
        """
        Sends text and images to Jina API and retrieves multi-vector matrices.
        """
        data = {
            "model": "jina-embeddings-v4",
            "task": "text-matching",
            "return_multivector": True,
            "input": inputs
        }
        response = requests.post(self.url, headers=self.headers, json=data).json()
        
        # Extract embeddings and convert to torch tensors for MaxSim calculation
        # Each entry is a list of vectors [[v1], [v2], ...]
        embeddings = [torch.tensor(item['embeddings']) for item in response['data']]
        return embeddings

    def maxsim(self, query_matrix, doc_matrix):
        """
        Calculates Late Interaction (MaxSim) similarity between two multi-vector matrices.
        """
        # Normalize vectors for cosine similarity
        query_matrix = F.normalize(query_matrix, p=2, dim=-1)
        doc_matrix = F.normalize(doc_matrix, p=2, dim=-1)
        
        # Compute all-to-all similarity [query_tokens, doc_tokens]
        sim_matrix = torch.matmul(query_matrix, doc_matrix.T)
        
        # MaxSim: For each query token, find the best match in the document
        max_scores = sim_matrix.max(dim=1)[0]
        return max_scores.mean().item()

    #with source, target image similarity
    def compute_mmss_advanced(self, source_en, candidate_de, image_url):
        # 1. Get all embeddings
        inputs = [{"text": source_en}, {"text": candidate_de}, {"image": image_url}]
        embs = self.get_multivectors(inputs)
        
        en_vec, de_vec, img_vec = embs[0], embs[1], embs[2]

        # 2. Compute Raw Similarities
        # How well does the Candidate match the Source? (Textual Fidelity)
        sim_text = self.maxsim(de_vec, en_vec)
        
        # How well does the Candidate match the Image? (Candidate Grounding)
        sim_cand_img = self.maxsim(de_vec, img_vec)
        
        # [NEW] How well does the Source match the Image? (Baseline Grounding)
        sim_src_img = self.maxsim(en_vec, img_vec)

        # 3. Intelligent Scoring Logic
        
        # A. Semantic Preservation Score (The classic text score)
        score_text = sim_text
        
        # B. Visual Preservation Score (Relative Grounding)
        # We compare the Candidate's visual score against the Source's visual score.
        # If Source score is low (<0.3), the image is likely irrelevant/noisy.

        lambda_weight = max(0, sim_src_img) ** 2
        
        mmss_rwf = (sim_text + (lambda_weight * sim_cand_img)) / (1 + lambda_weight)
        
        if sim_src_img < 0.3:
            # Image is bad/irrelevant. Fallback to text-only score to avoid noise.
            print(f"Warning: Low source-image alignment ({sim_src_img:.3f}). Ignoring visual score.")
            final_score = score_text
            visual_contribution = 0.0
        else:
            # Image is valid.
            # We want the candidate to achieve at least 90% of the source's visual score.
            # We minimize the "Visual Drop" (Source_Vis - Cand_Vis)
            visual_drop = max(0, sim_src_img - sim_cand_img)
            
            # Penalize the text score based on how much visual info was lost
            visual_penalty = visual_drop * 1.5 # Weighting factor
            final_score = score_text - visual_penalty
            visual_contribution = sim_cand_img

        return {
            "MMSS_Advanced": round(mmss_rwf, 4),
            "Text_Fidelity": round(sim_text, 4),
            "Visual_Ref": round(sim_src_img, 4),  # The Baseline
            "Visual_Cand": round(sim_cand_img, 4) # The Candidate
        }
    def compute_mmss(self, source_en, candidate_de, image_url):
        # 1. Get embeddings for all three modalities in one API call
        inputs = [{"text": source_en}, {"text": candidate_de}, {"image": image_url}]
        embs = self.get_multivectors(inputs)
        
        en_vec, de_vec, img_vec = embs[0], embs[1], embs[2]

        # 2. Calculate scores
        # How well does the German match the English?
        textual_fidelity = self.maxsim(de_vec, en_vec)
        
        # How well does the German match the Image?
        visual_grounding = self.maxsim(de_vec, img_vec)

        # MMSS: Balanced score
        mmss = 2 * (textual_fidelity * visual_grounding) / (textual_fidelity + visual_grounding + 1e-9)

        return {
            "MMSS": round(mmss, 4),
            "Fidelity": round(textual_fidelity, 4),
            "Grounding": round(visual_grounding, 4)
        }

# --- Execution ---
API_KEY = "jina_981e851b2dee47ba834256269776c26dF2quAX8oYUwN-8M7_jQapBtD-9As"
scorer = JinaAPIv4Scorer(API_KEY)

results = scorer.compute_mmss_advanced(
    source_en="A beautiful sunset over the beach",
    candidate_de="Ein schöner Sonnenuntergang am Strand",
    image_url="https://i.ibb.co/r5w8hG8/beach2.jpg"
)

print(results)

results = scorer.compute_mmss(
    source_en="A beautiful sunset over the beach",
    candidate_de="Ein schöner Sonnenuntergang am Strand",
    image_url="https://i.ibb.co/r5w8hG8/beach2.jpg"
)   

print(results)
