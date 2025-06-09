from transformers import AutoModel
import numpy as np


# Initialize the model
model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)

# New meaningful sentences
#sentences = ['Un chat bleu', 'A blue cat']

sentences = ['a calico cat is cuddling with an orange dog on a blanket.', 'a grey cat is cuddling with an orange cat on a blanket.', 'Eine graue Katze kuschelt mit einer orangefarbenen Katze auf einer Decke.']

# Public image URLs
#image_urls = [
#    'https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg', #Blue cat image
#    'https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg' #Red cat image
#]

image_urls = ['image1.jpg']
# Encode text and images
text_embeddings = model.encode_text(sentences)
image_embeddings = model.encode_image(image_urls)  # also accepts PIL.image, local filenames, dataURI

# Compute similarities

#Cosine Similarity of text-text and text-image
#combinding the similarity of both

# pseudo code

candiate_reference= text_embeddings[0] @ text_embeddings[1].T
candidate_image = text_embeddings[0] @ image_embeddings[0].T
ref_images = text_embeddings[1] @ image_embeddings[0].T

cand_source = text_embeddings[0] @ text_embeddings[2].T
source_images = text_embeddings[2] @ image_embeddings[0].T

print("cand-refere", candiate_reference)
print("cand-image", candidate_image)
print("ref-image", ref_images)



print("cand-source", cand_source)
print("source_images", source_images)


#Weighted joint similairty
weights=(0.2, 0.4, 0.4)
w1, w2, w3 = weights
assert abs(w1 + w2 + w3 - 1.0) < 1e-6, "Weights must sum to 1"
  
j_s = w1 * candiate_reference + w2 * candidate_image + w3 * ref_images
print("wighted average_cand_ref-cand_image-ref_image", j_s)

j_s_r_image = w1 * candiate_reference + w2 * cand_source+ w3 * source_images
print("wighted average-cand_ref-cand_src-src_img", j_s_r_image)

weights=(0.33, 0.33, 0.33)
w1, w2, w3 = weights
j_s_r_image = w1 * candidate_image + w2 * cand_source+ w3 * source_images
print("wighted average_src-imge-cand-src-src-img", j_s_r_image)

#print(text_embeddings[0] @ text_embeddings[1].T) # text embedding similarity
#print(text_embeddings[0] @ image_embeddings[0].T) # text-image cross-modal similarity
#print(text_embeddings[0] @ image_embeddings[1].T) # text-image cross-modal similarity
#print(text_embeddings[1] @ image_embeddings[0].T) # text-image cross-modal similarity
#print(text_embeddings[1] @ image_embeddings[1].T)# text-image cross-modal similarity