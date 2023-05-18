import models 

def Model_Architecture(state):

    if state.MODEL_ARCHITECTURES[state.args.m] == 'Lenet5':
        model = models.LeNet()
    elif state.MODEL_ARCHITECTURES[state.args.m] == 'VGG-16' and state.args.d !=0:
        model = models.VGG('VGG11',ch=3)
    elif state.MODEL_ARCHITECTURES[state.args.m] == 'VGG-16' and state.args.d ==0:
        model = models.VGG('VGG11',ch=1) 
    elif state.MODEL_ARCHITECTURES[state.args.m] == 'ResNet-18':
        num_classes = 10
        if state.args.d == 2:
            num_classes=1000
        model = models.resnet.ResNet18(num_classes)
    elif state.MODEL_ARCHITECTURES[state.args.m] == "EfficientNet_B0":
        if state.args.d == 2:
            model = models.efficientnet.EfficientNetB0(1000)
        else:
            model = models.efficientnet.EfficientNetB0(10)
    elif state.MODEL_ARCHITECTURES[state.args.m] == "LocalVit":
        model = models.localvit.create_model_localvit()#TODO:
    return model