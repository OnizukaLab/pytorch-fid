# -*- coding: utf-8 -*-
# # OverView
# 正解画像と正例画像の画像特徴量の近さを見る．ただし画像にぼかしを適用し情報を落とす．様々な強さのぼかしでこれを行う．
# ぼかしが強いほどマッチしやすいと考えられる．いい生成画像ほど，ぼかしが弱くても良くマッチすると考えられる
# # Name
# Multi Level Blurred  Image Match（仮）
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import ImageFilter
from torchvision.models.inception import inception_v3


class BlurredImagesDataset(Dataset):

    def __init__(self, real_image, fake_image, blur_level=5, img_size=256):


if __name__ == '__main__':
