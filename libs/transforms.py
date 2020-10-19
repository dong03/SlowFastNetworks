import solt
import solt.transforms as slt
from albumentations.pytorch.functional import img_to_tensor

def create_train_transforms(size):
    return solt.Stream([
                        slt.JPEGCompression(p=0.5,quality_range=(60,100)),
                        slt.Noise(p=0.25),
                        slt.Brightness(),
                        slt.Contrast(),
                        slt.Flip(),
                        slt.Rotate90(),
                        solt.SelectiveStream([
                            slt.GammaCorrection(gamma_range=0.5, p=1),
                            slt.Noise(gain_range=0.1, p=1),
                            slt.SaltAndPepper(),
                            slt.Blur(),
                        ], n=3),
                        slt.Rotate(angle_range=(-10, 10), p=0.5),
                        slt.Resize((size,size)),
                    ])

def create_val_transforms(size):
    return solt.Stream([slt.Resize((size,size))])

def direct_val(imgs,size):
    #img 输入为RGB顺序
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    stream = create_val_transforms(size)
    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}
    imgs = [img_to_tensor(stream(each,return_torch=False).data[0],normalize).unsqueeze(0) for each in imgs]
    imgs = torch.cat(imgs)

    return imgs

def reverse_transform(inputs):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i,m,s in zip(inputs,mean,std):
        i.mul_(s).add_(m)
    outputs = inputs.detach().cpu()
    imgs = [np.array(transforms.ToPILImage()(outputs[i])) for i in range(inputs.shape[0])]
    return imgs