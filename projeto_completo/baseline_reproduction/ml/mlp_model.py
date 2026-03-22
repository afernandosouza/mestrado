from sklearn.neural_network import MLPClassifier


def create_mlp():

    return MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="tanh",
        solver="adam",
        max_iter=2000,
        random_state=None
    )