HYPERPARAMETERS = {
    "batch_size": [32, 128, 64],
    "learning_rate": [0.1, 0.05, 0.01, 0.001],
    "weight_decay": [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9, 0.8, 0.5],
    "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
    "pos_weight" : [1.0, 1.3, 0.8, 0.9],  
    "model_embedding_size": [8, 16, 32, 64, 128],
    "model_attention_heads": [1, 2, 3, 4],
    "model_layers": [2, 4, 5, 8, 10],
    "model_dropout_rate": [0.2, 0.5, 0.9],
    "model_top_k_ratio": [0.2, 0.5, 0.8, 0.9],
    "model_top_k_every_n": [1, 2, 3],
    "model_dense_neurons": [128, 64, 256, 32]
}

BEST_PARAMETERS = {
    "batch_size": [128],
    "learning_rate": [	0.05],
    "weight_decay": [0.00001],
    "sgd_momentum": [0.5],
    "scheduler_gamma": [0.995],
    "pos_weight": [1],
    "model_embedding_size": [64],
    "model_attention_heads": [3],
    "model_layers": [8],
    "model_dropout_rate": [0.5],
    "model_top_k_ratio": [0.9],
    "model_top_k_every_n": [3],
    "model_dense_neurons": [256]
}