manual_themes_15 = [
   <put list of themes here> 
]


golden_themes_12 = [
   <put list of themes here>
]


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------
# Step 2: Load embedding model
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode both sets
manual_embeddings = model.encode(manual_themes_15, convert_to_tensor=True)
golden_embeddings = model.encode(golden_themes_12, convert_to_tensor=True)

# ---------------------------
# Step 3: Compute cosine similarity
# ---------------------------
similarity_matrix = cosine_similarity(manual_embeddings, golden_embeddings)

# Convert to DataFrame for labeling
df_sim = pd.DataFrame(similarity_matrix, 
                      index=manual_themes_15, 
                      columns=golden_themes_12)

# ---------------------------
# Step 4: Plot heatmap
# ---------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(df_sim, 
            annot=True, fmt=".2f", 
            cmap='viridis', 
            cbar_kws={'label': 'Similarity Score'})

plt.title("Cosine Similarity Heatmap (Manual vs Golden Themes)", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
