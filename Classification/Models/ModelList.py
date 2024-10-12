from typing import List


def architecture() -> list:
    arch: List[str] = [
            'AlexNet',
            'ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large',
            'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet169', 'DenseNet201',
            'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2', 'EfficientNet_B3',
            'EfficientNet_B4', 'EfficientNet_B5', 'EfficientNet_B6', 'EfficientNet_B7',
            'EfficientNetV2_S', 'EfficientNetV2_M', 'EfficientNetV2_L',
            'GoogLeNet',
            'InceptionV3', 'InceptionV4',
            'MaxVit_T',
            'MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3',
            'MobileNetV2',
            'MobileNetV3_Large', 'MobileNetV3_Small',
            'RegNetY_400mf', 'RegNetY_800mf',
            'RegNetY_1_6GF', 'RegNetY_3_2GF', 'RegNetY_8gf', 'RegNetY_16gf', 'RegNetY_32gf', 'RegNetY_128GF',
            'RegNetX_400mf', 'RegNetX_800mf',
            'RegNetX_1_6GF', 'RegNetX_3_2GF', 'RegNetX_8gf', 'RegNetX_16gf', 'RegNetX_32gf',
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'ResNeXt50_32x4d', 'ResNeXt101_32x8d', 'ResNeXt101_64x4d',
            'ShuffleNetV2_X0_5', 'ShuffleNetV2_X1_0', 'ShuffleNetV2_X1_5', 'ShuffleNetV2_X2_0',
            'SqueezeNet1_0', 'SqueezeNet1_1',
            'SwinTransformer_T', 'SwinTransformer_S', 'SwinTransformer_B',
            'SwinTransformerV2_T', 'SwinTransformerV2_S', 'SwinTransformerV2_B',
            'VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN',
            'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14',
            'WideResNet50_2', 'WideResNet101_2'
    ]
    return arch
