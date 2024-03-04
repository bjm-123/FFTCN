# %%
class Parameter:
    fs = 100
    n_class = 5

    cnn = [
        {'in_channels': 1, 'out_channels': 128, 'kernel_size': 50, 'stride': 25, 'padding': 0},
        {'kernel_size': 8, 'stride': 8, 'padding': 0},
        {'p': 0.5},
        {'in_channels': 128, 'out_channels': 128, 'kernel_size': 8, 'stride': 1, 'padding': 3},
        {'in_channels': 128, 'out_channels': 128, 'kernel_size': 8, 'stride': 1, 'padding': 3},
        {'in_channels': 128, 'out_channels': 128, 'kernel_size': 8, 'stride': 1, 'padding': 3},
        {'kernel_size': 4, 'stride': 4, 'padding': 0},
        {'p': 0.5},
    ]

    feature_net = {
        'cnn': cnn,
        'classifier': [
            {'in_channels': 128, 'out_channels': n_class, 'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'output_size': (1)}
        ]
    }

    sleep_net = {
        'feature_net': feature_net,
        'lstm': {
            'input_size': 256,
            'hidden_size': 128,
            'num_layers': 1,
            'bidirectional': True,
        },
        'tcn': {
            'in_channle': 256,
            'num_channels': [128, 128, 128, 128],
            'kernel_size': 3,
            'dropout': 0.2,
            'causal': False,
            'channle_last': True,
        },
        'classifier': {
            'in_features': 128,
            'out_features': n_class
        }
    }
