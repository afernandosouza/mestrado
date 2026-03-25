# src/ml/mlp_model.py

from sklearn.neural_network import MLPClassifier

def create_mlp(random_state=None):
    """
    Cria MLP reproduzindo exatamente as especificações do artigo:
    - 32 neurônios ocultos
    - tanh activation
    - Scaled Conjugate Gradient (solver='lbfgs' aproxima)
    - max_iter=5000
    """
    return MLPClassifier(
        hidden_layer_sizes=(32,),      # 32 neurônios ocultos
        activation='tanh',             # tanh (artigo)
        solver='lbfgs',                # Aproxima Scaled Conjugate Gradient
        alpha=1e-4,                    # Regularização
        max_iter=5000,                 # Máx épocas (artigo)
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2        # 80/20 split (artigo)
    )