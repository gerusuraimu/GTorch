from typing import List


def architecture() -> list:
    arch: List[str] = [
            'AlexNet',

            'ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large',

            'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet169', 'DenseNet201',

            'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2', 'EfficientNet_B3',
            'EfficientNet_B4', 'EfficientNet_B5', 'EfficientNet_B6', 'EfficientNet_B7',
            'EfficientNet_V2_S', 'EfficientNet_V2_M', 'EfficientNet_V2_L',

            'GoogLeNet',

            'Inception_V3',

            'MaxVit_T',

            'MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3',

            'MobileNet_V2',
            'MobileNet_V3_Large', 'MobileNet_V3_Small',

            'RegNet_Y_400mf', 'RegNet_Y_800mf',
            'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_8gf', 'RegNet_Y_16gf', 'RegNet_Y_32gf', 'RegNet_Y_128GF',
            'RegNet_X_400mf', 'RegNet_X_800mf',
            'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_8gf', 'RegNet_X_16gf', 'RegNet_X_32gf',

            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',

            'ResNeXt50_32x4d', 'ResNeXt101_32x8d', 'ResNeXt101_64x4d',

            'ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0',

            'Swin_T', 'Swin_S', 'Swin_B',
            'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B',

            'VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN',

            'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14',

            'Wide_ResNet50_2', 'Wide_ResNet101_2'
    ]
    return arch
