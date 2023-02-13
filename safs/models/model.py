import settings as settings
import models.dense_model_architecture as dense_model_arch
import models.pruned_model_architecture as pruned_model_arch
import model as model_

def Model_Architecture():

    if settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'Lenet5':
        model = dense_model_arch.LeNet5()
    elif settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'VGG-16':
        model = model_.VGG('VGG16')
    elif settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == 'ResNet-18':
        model = model_.resnet.ResNet18()
    elif settings.MODEL_ARCHITECTURES[settings.MODEL_ARCHITECTURE] == "EfficientNet_B0":
        model = model_.efficientnet_test.EfficientNetB0()
    return model