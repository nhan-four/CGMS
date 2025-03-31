import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Input, Dense, Dot, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def attention_mechanism(inputs):
    query = Dense(40)(inputs)
    key = Dense(40)(inputs)
    value = Dense(40)(inputs)

    scores = Dot(axes=-1)([query, key])
    scores = Activation('softmax')(scores)

    weighted_values = Dot(axes=1)([scores, value])
    return weighted_values

def create_dnn_attention_model(input_dim):
    input_shape = (1, input_dim)
    input_embedding = Input(shape=input_shape)
    attention_output = attention_mechanism(input_embedding)

    attention_output = Flatten()(attention_output)

    x = Dense(64, activation='relu')(attention_output)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_embedding, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train_augmented, X_test, y_test, epochs=50, batch_size=32):
    
    model.fit(X_train, y_train_augmented, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int) 

    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    cr = classification_report(y_test, y_pred_classes, digits=4)
    print("Classification Report:")
    print(cr)