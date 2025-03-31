import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

def create_transformer_model(output_dim):
    input_shape = (None, output_dim)
    inputs = Input(shape=input_shape)

    attention_output = MultiHeadAttention(num_heads=8, key_dim=output_dim)(inputs, inputs)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ffn_output = Dense(512, activation='relu')(attention_output)
    ffn_output = Dropout(0.1)(ffn_output)
    ffn_output = Dense(output_dim)(ffn_output)  

    encoder_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    flattened_output = Flatten()(encoder_output)

    transformer_model = Model(inputs=inputs, outputs=flattened_output)
    return transformer_model

def get_embedding(data_scaled):
    data_reshaped = data_scaled 
    output_dim = data_reshaped.shape[1] 
    transformer_model = create_transformer_model(output_dim)

    data_reshaped_expanded = np.expand_dims(data_reshaped, axis=1) 

    embedding = transformer_model.predict(data_reshaped_expanded)

    embedding_df = pd.DataFrame(embedding)  
    return embedding_df