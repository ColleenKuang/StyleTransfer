import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy

def loadimg(path=None):
    transform = transforms.Compose([transforms.Scale([224, 224]),
                                    transforms.ToTensor()])
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def img_show(img, title=None):
    img = img.clone().cpu()
    img = img.view(3, 224, 224)
    ToPIL = torchvision.transforms.ToPILImage()
    img = ToPIL(img)

    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.show()

'''
input: the content image
target：the content(feature) from original image(content img) learned from CNN
weight：a ratio factor affects the style and content to the original image 
loss: use MSE to compute
'''
class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_fn(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


'''
Gram矩阵是矩阵的内积运算，
运算过后特征图中越大的数字会变得更大，
相当于对图像的特性进行了缩放，
使得特征突出了，也就相当于提取到了图片的风格。
'''
class gram_matrix(torch.nn.Module):
    def forward(self, input):
        a,b,c,d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a*b*c*d)

'''
input: the content image
target：the content(feature) from original image(content img) learned from CNN
weight：a ratio factor affects the style and content to the original image 
loss: use MSE to compute
gram: do the gram computation based on featured map
'''
class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = gram_matrix()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.loss_fn(self.G, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


if __name__=='__main__':
    #Input images
    content_img = loadimg("images/4.jpg")
    style_img = loadimg("images/2.jpg")
    print(content_img.size())

    #display original picture and style picture
    img_show(content_img.data, title = "Content_Image")
    img_show(style_img.data, title = "Style_Image")

    #detect gpu device,
    use_gpu = torch.cuda.is_available()

    #download pretrained model vgg16
    cnn = models.vgg16(pretrained=True).features
    if use_gpu:
        cnn = cnn.cuda()

    content_layer = ["Conv_5","Conv_6"]
    style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]

    content_losses = []
    style_losses = []

    conten_weight = 1
    style_weight = 1000

    #Transfer Learning
    new_model = torch.nn.Sequential()
    model = copy.deepcopy(cnn)
    gram = gram_matrix()

    if use_gpu:
        new_model = new_model.cuda()
        gram = gram.cuda()

    index = 1
    for layer in list(model):
        if isinstance(layer, torch.nn.Conv2d):
            name = "Conv_" + str(index)
            new_model.add_module(name, layer)
            if name in content_layer:
                target = new_model(content_img).clone()
                content_loss = Content_loss(conten_weight, target)
                new_model.add_module("content_loss_" + str(index), content_loss)
                content_losses.append(content_loss)

            if name in style_layer:
                target = new_model(style_img).clone()
                target = gram(target)
                style_loss = Style_loss(style_weight, target)
                new_model.add_module("style_loss_" + str(index), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, torch.nn.ReLU):
            name = "Relu_" + str(index)
            new_model.add_module(name, torch.nn.ReLU(inplace=False))
            index = index + 1

        if isinstance(layer, torch.nn.MaxPool2d):
            name = "MaxPool_" + str(index)
            new_model.add_module(name, layer)
    print(new_model)

    input_img = content_img.clone()

    parameter = torch.nn.Parameter(input_img.data)
    optimizer = torch.optim.LBFGS([parameter])

    #Training
    n_epoch = 1000
    run = [0]
    while run[0] <= n_epoch:
        def closure():
            optimizer.zero_grad()
            style_score = 0
            content_score = 0
            parameter.data.clamp_(0, 1)
            new_model(parameter)
            for sl in style_losses:
                style_score += sl.backward()

            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print('{} Style Loss : {:4f} Content Loss: {:4f}'.format(run[0],
                                                                         style_score.data, content_score.data))
            return style_score + content_score


        optimizer.step(closure)

    #Display result
    parameter.data.clamp_(0,1)
    img_show(parameter.data, title="Output Image")