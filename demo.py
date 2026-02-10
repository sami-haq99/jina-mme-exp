import os
import base64
from similarity_score import JinaV4SimilarityMapper

def save_base64_image(base64_str, output_path):
    """Helper to decode base64 string and save as a PNG file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_str))
    print(f"Saved: {output_path}")

def main():
    # 1. Initialize the Mapper
    # Use client_type="web" if you don't have the model locally and want to use the API class provided
    # Use client_type="local" if you have the model weights and want to run it on GPU
    # Note: For "web" mode, you might need to set an API key in the JinaEmbeddingsClient class or passing it if modified.
    # The provided code has "Bearer Not Set" by default.
    #task = [retrieval, text-matching]
    task = 'text-matching'
    mapper = JinaV4SimilarityMapper(task=task)  # or "local" if you have the model

    # 2. Define Inputs
    # You can use a local file path or a URL
    image_source = "https://cdn.duvine.com/wp-content/uploads/2016/04/17095703/Slides_mallorca_FOR-WEB.jpg" 
    text_query = "A group of cats walking nearby the ocean"
    #img_proc, *_ = mapper.process_image(image_source)    
    # Create output directory
    output_dir = f"heatmap_results_{task}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing query: '{text_query}'")
    print("Generating heatmaps...")

    # 3. Generate Heatmaps
    try:
        tokens, heatmaps, g_score = mapper.get_token_similarity_maps(
            query=text_query,
            image=image_source
        )

        # 4. Save Results
        
        #save g_score in a text file
        with open(os.path.join(output_dir, f"g_{task}_score.txt"), "w") as f:
            f.write(str(g_score))

        print(f"\nFound {len(tokens)} valid tokens_score.", g_score)
        for token in tokens:
            if token in heatmaps:
                # Create a safe filename for the token
                safe_token_name = "".join([c if c.isalnum() else "_" for c in token])
                filename = f"heatmap_{safe_token_name}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Decode and save
                save_base64_image(heatmaps[token], output_path)

        print("\nAll heatmaps saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()