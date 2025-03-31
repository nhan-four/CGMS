from Lib import *
from data_pre import prepare_data
from trans_model import get_embedding
from IP_process import process_embedding
from CG_train import train_cg, generate_data
from dnn_atten import create_dnn_attention_model, train_and_evaluate_model

if __name__ == "__main__":
    file_path = 'data_sorted.csv'
    data, data_reshaped = prepare_data(file_path)

    embedding_df = get_embedding(data_reshaped)

    ip_pair_path = 'data_IPPair.csv'
    output_path = 'final_vectors_transformer_real.csv'
    final_vectors = process_embedding(embedding_df, ip_pair_path, output_path)

    labeled_data = pd.read_csv('IP_couple_index.csv', index_col=0)
    label_encoder = LabelEncoder()
    labeled_data['Pair_Labeled'] = label_encoder.fit_transform(labeled_data['Pair_Label'])
    y = labeled_data['Pair_Labeled'].values

    scaler = StandardScaler()
    final_vectors_scaled = scaler.fit_transform(final_vectors)
    X = pd.DataFrame(final_vectors_scaled, columns=final_vectors.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.values

    labels = data['Label'].values
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
    label_conditions = np.array([label_mapping[label] for label in labels])
    minority_indices = np.where(y_train == 1)[0]
    data_minority = X_train[minority_indices]
    label_conditions = y_train[minority_indices]

    latent_dim = 100
    batch_size = 128
    epochs = 300
    generator = train_cg(
        X_train, y_train, latent_dim=latent_dim, batch_size=batch_size, epochs=epochs
    )

    X_train_augmented, y_train_augmented = generate_data(
        generator, X_train, y_train, latent_dim=latent_dim
    )

    X_train = np.reshape(X_train_augmented, (X_train_augmented.shape[0], 1, X_train_augmented.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    input_dim = int(X_train.shape[2])
    model = create_dnn_attention_model(input_dim)
    train_and_evaluate_model(
        model, X_train, y_train_augmented, X_test, y_test, epochs=50, batch_size=32
    )
