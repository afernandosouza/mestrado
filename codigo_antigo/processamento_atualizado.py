import numpy as np
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def text_to_series(text):
    """Converte um texto em uma série temporal baseada em UTF-8."""
    return np.array([ord(c) for c in text], dtype=np.float32)

def wavelet_decomposition(series, wavelet='db4', level=5):
    """Aplica decomposição wavelet e extrai 32 características de cada subbanda."""
    coeffs = pywt.wavedec(series, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])  # Extraindo média e desvio padrão
    return features[:32]  # Limitando a 32 features

def prepare_dataset(texts, n_clusters=5):
    """Converte textos em séries, agrupa em clusters e extrai características."""
    series = [text_to_series(text) for text in texts]
    features = np.array([wavelet_decomposition(s) for s in series])
    
    # Normalização
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    return features_scaled, labels, kmeans

def build_mlp(input_dim):
    """Cria um modelo de rede neural perceptron multicamadas."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')  # Supondo 5 clusters
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Exemplo de uso
data = ["Olá, mundo!", "Hello, world!", "Bonjour le monde!", "Hallo Welt!", "Hola mundo!"]
X, y, kmeans = prepare_dataset(data)
model = build_mlp(X.shape[1])

# Treinamento
epochs = 50
history = model.fit(X, y, epochs=epochs, verbose=1, validation_split=0.2)

# Exibir gráfico de clusters
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title('Clusters dos Textos')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Exibir gráfico de perda e precisão durante o treinamento
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Perda (Treino)')
plt.plot(history.history['val_loss'], label='Perda (Validação)')
plt.title('Evolução da Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Acurácia (Treino)')
plt.plot(history.history['val_accuracy'], label='Acurácia (Validação)')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
