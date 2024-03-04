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

    raw_feature_net = {
        'cnn': cnn,
        'classifier': [
            {'in_channels': 128, 'out_channels': n_class, 'kernel_size': 2, 'stride': 1, 'padding': 0},
            {'output_size': (1)}
        ]
    }

    vgg = [
        {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 32, 'out_channels': 48, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 48, 'out_channels': 48, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 48, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 64, 'out_channels': 72, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 72, 'out_channels': 72, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        ##
        {'in_channels': 72, 'out_channels': 72, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'kernel_size': 1, 'stride': 1, 'padding': 0},
    ]

    wave_feature_net = {
        'vgg': vgg,
        'classifier': [
            {'in_channels': 72, 'out_channels': n_class, 'kernel_size': (1, 3), 'stride': (1, 1), 'padding': (0, 0)},
            {'output_size': (1, 1)}
        ]
    }

    sleep_net = {
        'raw_feature_net': raw_feature_net,
        'wave_feature_net': wave_feature_net,
        'lstm': {
            'input_size': 1344 + 256,
            'hidden_size': 128,
            'num_layers': 1,
            'bidirectional': True,
        },
        'tcn': {
            'in_channle': 472,  # 216+256,
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
