import pandas as pd

def process_embedding(embedding_df, ip_pair_path, output_path):
    IPPair = pd.read_csv(ip_pair_path)
    embedding_df['IPPair'] = IPPair['IPPair']
    grouped_embedding = embedding_df.groupby('IPPair').median()
    final_vectors = grouped_embedding.reset_index().drop(columns=['IPPair'])
    final_vectors.to_csv(output_path, index=False)
    return final_vectors