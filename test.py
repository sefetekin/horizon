import torch
import torchvision
import pdb
trainset = torchvision.datasets.ImageFolder(
    root='knee/train', 
    transform=torchvision.transforms.ToTensor()
        
) #this loads all the images and turns them into numbers so that computer can understand it

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

transforms_hflip = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=1)
])
transforms_affline = torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(degrees=0,shear=0.25)
])

image, labels = iter(train_dataloader).next()

import matplotlib.pyplot as plt
import numpy as np
def imshow(img, save=''):
    npimg = torchvision.utils.make_grid(img).numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    if len(save) > 0:
        plt.savefig(save)
    else: 
        plt.show()

# before any transformation
imshow(image, 'transformation_images/before.png')
# after hflip
imshow(transforms_hflip(image), 'transformation_images/afterhflip.png')
# after shear
imshow(transforms_affline(image),'transformation_images/aftershear.png')
with open('transformation_images/labels.txt', 'w') as f:
    
    f.write(str(labels))
