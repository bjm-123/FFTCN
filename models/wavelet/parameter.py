# %%
class Parameter:
    fs = 100
    n_class = 5

    # vgg = [
    #     {'in_channels':1, 'out_channels':32, 'kernel_size':(3, 5), 'stride':(1, 1), 'padding':(1, 0)},
    #     {'in_channels':32, 'out_channels':32, 'kernel_size':(3, 5), 'stride':(1, 1), 'padding':(1, 0)},
    #     {'kernel_size':2, 'stride':2, 'padding':0},
    #     {'in_channels':32, 'out_channels':48, 'kernel_size':(3, 5), 'stride':(1, 1), 'padding':(1, 0)},
    #     {'in_channels':48, 'out_channels':48, 'kernel_size':(3, 5), 'stride':(1, 1), 'padding':(1, 0)},
    #     {'kernel_size':2, 'stride':2, 'padding':0},
    #     {'in_channels':48, 'out_channels':64, 'kernel_size':(3, 5), 'stride':(1, 1), 'padding':(1, 0)},
    #     {'in_channels':64, 'out_channels':64, 'kernel_size':(3, 5), 'stride':(1, 1), 'padding':(1, 0)},
    #     {'kernel_size':2, 'stride':2, 'padding':0},
    # ]

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

    feature_net = {
        'vgg': vgg,
        'classifier': [
            {'in_channels': 72, 'out_channels': n_class, 'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0)},
            {'output_size': (1, 1)}
        ]
    }

    sleep_net = {
        'feature_net': feature_net,
        'lstm': {
            'input_size': 864,
            'hidden_size': 128,
            'num_layers': 1,
            'bidirectional': True,
        },
        'tcn': {
            'in_channle': 216,
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
