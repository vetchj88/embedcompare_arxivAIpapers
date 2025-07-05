import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---
# Choose which model's output you want to analyze.
# Options: 'papers_bge_large.json', 'papers_gte_large.json', 'papers_minilm.json'
DATA_FILE_TO_ANALYZE = 'papers_minilm.json'
NUM_TOP_KEYWORDS = 10 # How many keywords to show for each cluster

def analyze_cluster_topics():
    """
    Loads clustering data and uses TF-IDF to extract the defining
    keywords for each cluster based on the paper abstracts.
    """
    print("\n" + "="*55)
    print(f"  Topic Keyword Analysis for: {DATA_FILE_TO_ANALYZE}")
    print("="*55 + "\n")

    try:
        df = pd.read_json(DATA_FILE_TO_ANALYZE)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE_TO_ANALYZE}' was not found.")
        return

    # Group abstracts by cluster_id
    # We use a lambda function to handle potential None/NaN values in abstracts
    corpus_by_cluster = df.groupby('cluster_id')['abstract'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()

    # The documents for TF-IDF will be the combined abstracts of each cluster
    documents = corpus_by_cluster['abstract']
    
    # Use English stop words and set max_features to avoid overly rare words
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2) # Include bigrams (e.g., "language model")
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        print("Could not perform TF-IDF. This may be because there is only one cluster.")
        return
        
    feature_names = vectorizer.get_feature_names_out()

    for index, row in corpus_by_cluster.iterrows():
        cluster_id = row['cluster_id']
        
        # Get the TF-IDF vector for the current cluster
        tfidf_vector = tfidf_matrix[index]
        
        # Sort the terms by their TF-IDF score in descending order
        sorted_indices = tfidf_vector.toarray().argsort()[0, ::-1]
        
        # Get the top N keywords
        top_keywords = [feature_names[i] for i in sorted_indices[:NUM_TOP_KEYWORDS]]
        
        # Get number of papers in this cluster
        paper_count = len(df[df['cluster_id'] == cluster_id])
        
        cluster_label = f"Cluster {cluster_id}"
        if cluster_id == -1:
            cluster_label = "Outliers"
            
        print(f"--- {cluster_label} ({paper_count} papers) ---")
        print(", ".join(top_keywords))
        print("-" * (len(cluster_label) + 2 + len(str(paper_count)) + 10))
        print()

if __name__ == "__main__":
    analyze_cluster_topics()