# %%
class Parameter:
    fs = 100
    n_class = 5

    identity_encoder = [
        {'in_channels': 1, 'out_channels': 64, 'kernel_size': 50, 'stride': 25, 'padding': 0},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 8, 'stride': 1, 'padding': 0},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 64, 'out_channels': 80, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 80, 'out_channels': 96, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 96, 'out_channels': 112, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 112, 'out_channels': 128, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 128, 'out_channels': 144, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 144, 'out_channels': 160, 'kernel_size': 7, 'stride': 1, 'padding': 3},
    ]

    stage_encoder = [
        {'in_channels': 1, 'out_channels': 64, 'kernel_size': 50, 'stride': 25, 'padding': 0},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 8, 'stride': 1, 'padding': 0},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 64, 'out_channels': 80, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 80, 'out_channels': 96, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 96, 'out_channels': 112, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 112, 'out_channels': 128, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'kernel_size': 2, 'stride': 2, 'padding': 0},
        {'in_channels': 128, 'out_channels': 144, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 144, 'out_channels': 160, 'kernel_size': 7, 'stride': 1, 'padding': 3},
    ]

    decoder = [
        {'in_channels': 320, 'out_channels': 288, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 288, 'out_channels': 256, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'scale_factor': 2},
        {'in_channels': 256, 'out_channels': 224, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 224, 'out_channels': 192, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'scale_factor': 2},
        {'in_channels': 192, 'out_channels': 160, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'in_channels': 160, 'out_channels': 128, 'kernel_size': 7, 'stride': 1, 'padding': 3},
        {'scale_factor': 2},
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 8, 'stride': 1, 'padding': 0},
        {'in_channels': 64, 'out_channels': 1, 'kernel_size': 50, 'stride': 25, 'padding': 0},
    ]

    classifier = {
        'channel_in': 2240,
        'channels': [64, 64, 64],
        'kernel_size': 7,
        'dropout': 0.2,
        'n_class': n_class,
    }
