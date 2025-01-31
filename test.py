from PIL import Image
import numpy as np
import os
import torch

import time
import imageio

import torchvision.transforms as transforms

from Networks.Mymodules import MODEL as net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')



model = net(in_channel=2)

model  = model.to(device)

model_path = "models//model.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()
    device_ids = range(torch.cuda.device_count())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(model_path))
    print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)


def fusion_gray():

    for num in range(1, 101):

        path1 = './source images/underexposed image.bmp '

        path2 = './source images/overexposed image.bmp '
        imgA = Image.open(path1).convert('L')

        imgB = Image.open(path2).convert('L')


        img1_read = np.array(imgA)

        imgA1_org = imgA

        imgB1_org = imgB


        tran = transforms.ToTensor()

        imgA_org = tran(imgA1_org)

        imgB_org = tran(imgB1_org)

        input_img1 = torch.cat((imgA_org, imgB_org), 0).unsqueeze(0)

        if use_gpu:
            input_img1 = input_img1.cuda()
        else:
            input_img1 = input_img1


        model.eval()
        out = model(input_img1)

        d_map_1= np.squeeze(out.detach().cpu().numpy())

        decision_1_4 = (d_map_1 * 255).astype(np.uint8)

        imageio.imwrite('./fusion result/fused image.bmp',
                        decision_1_4)





if __name__ == '__main__':

    fusion_gray()
