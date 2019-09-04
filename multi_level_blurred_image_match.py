# -*- coding: utf-8 -*-
# # OverView
# 正解画像と正例画像の画像特徴量の近さを見る．ただし画像にぼかしを適用し情報を落とす．様々な強さのぼかしでこれを行う．
# ぼかしが強いほどマッチしやすいと考えられる．いい生成画像ほど，ぼかしが弱くても良くマッチすると考えられる
# # Name
# Multi Level Blurred  Image Match（仮）
from argparse import ArgumentParser
import pathlib
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from inception import InceptionV3
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help='Required. Path to dataset having subdirectories "train" and "test".')
    parser.add_argument("--fake", type=str, required=True,
                        help="Required. Path to images the model generated.")
    parser.add_argument("--gpu", type=int, required=False, default=-1,
                        help="Optional. GPU id to use. If not specified, no GPU will be used (CPU only).")
    parser.add_argument("--split", type=int, required=False, default=10,
                        help="Optional. Default 10. The number the test set will be split into when aggregation.")
    parser.add_argument("--num_workers", type=int, required=False, default=4,
                        help="Optional. num_workers for data loader.")
    return parser.parse_args()


class BlurredImagesDataset(Dataset):

    def __init__(self, data_root, fake_image, img_size=256):
        self.data_root = pathlib.Path(data_root)
        self.fake_image_path = pathlib.Path(fake_image)
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.blur0 = transforms.Lambda(
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=img_size/(2**4))))  # strongest blur
        self.blur1 = transforms.Lambda(
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=img_size/(2**5))))
        self.blur2 = transforms.Lambda(
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=img_size/(2**6))))
        self.blur3 = transforms.Lambda(
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=img_size/(2**7))))
        self.blur4 = transforms.Lambda(
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=img_size/(2**8))))
        self.blur5 = transforms.Lambda(
            lambda im: im.filter(ImageFilter.GaussianBlur(radius=img_size/(2**9))))  # weakest blur

        self.norm = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = 10
        with open(self.data_root / "test/filenames.pickle", "rb") as f:
            self.test_filenames = pickle.load(f, encoding="latin1")
        self.bbox = self.load_bbox()

    def load_bbox(self):
        bbox_path = self.data_root / 'CUB_200_2011/bounding_boxes.txt'
        file_path = self.data_root / 'CUB_200_2011/images.txt'
        with open(str(bbox_path), "r") as f:
            bbox_list = [[float(e) for e in line.strip().split()] for line in f]
        with open(str(file_path), "r") as f:
            file_list = [line.strip().split()[1][:-4] for line in f]
        file_bbox = {k: v for k, v in zip(file_list, bbox_list)}
        bbox_list = [file_bbox[file] for file in self.test_filenames]
        return bbox_list

    def get_image(self, path, index, bbox=None):
        image_path = path / (self.test_filenames[index//self.embeddings_num]+".jpg")
        if not image_path.exists():
            image_path = list(
                path.glob(
                    self.test_filenames[index//self.embeddings_num] + "*{}.jpg".format(index % self.embeddings_num))
            ) + list(path.glob("{}.jpg".format(index)))\
                         + list(path.glob("{}_fake_*".format(index)))
            assert len(image_path) == 1, "Invalid image path {}".format("\n".join([str(p) for p in image_path]))
            image_path = image_path[0]
        image = Image.open(image_path).convert("RGB")
        if bbox is not None:
            width, height = image.size
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            image = image.crop([x1, y1, x2, y2])
        return image

    def __getitem__(self, index):
        real = self.get_image(self.data_root/"CUB_200_2011/images", index, self.bbox[index//self.embeddings_num])
        fake = self.get_image(self.fake_image_path, index)
        real_set = []
        fake_set = []
        for blur in self.blur0, self.blur1, self.blur2, self.blur3, self.blur4, self.blur5:
            real_set.append(self.norm(blur(real)))
            fake_set.append(self.norm(blur(fake)))
        return real_set + [self.norm(real)] + fake_set + [self.norm(fake)]

    def __len__(self):
        return len(self.test_filenames)


# def collate_fn(image_sets):
#     real_dict = {i: [] for i in range(7)}
#     fake_dict = {i: [] for i in range(7)}
#     for image_set in image_sets:
#         for i in range(7):
#             real_img, fake_img = image_set[i]
#             real_dict[i].append(real_img)
#             fake_dict[i].append(fake_img)
#     mini_batch = [(torch.stack(real_dict[i])+torch.stack(fake_dict[i])) for i in range(7)]
#     return mini_batch


if __name__ == '__main__':
    args = parse_args()
    dataset = BlurredImagesDataset(args.data_root, args.fake)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, drop_last=False, num_workers=args.num_workers)
    device = torch.device("cuda:{}".format(args.gpu) if (args.gpu != -1 and torch.cuda.is_available()) else "cpu")
    inception_model = InceptionV3(output_blocks=[3], normalize_input=False)
    inception_model.to(device)
    inception_model.eval()

    scores = {i: [] for i in range(7)}
    criterion = torch.nn.MSELoss(reduction="none")
    for data in tqdm(dataloader):
        reals, fakes = data[:7], data[7:]
        for i in range(7):
            real, fake = reals[i], fakes[i]
            real = real.to(device)
            fake = fake.to(device)
            real_feature = inception_model(real)[0]
            fake_feature = inception_model(fake)[0]
            score = criterion(real_feature, fake_feature)
            scores[i] += score.detatch().tolist()
    split_batch_size = len(dataset)//args.split
    for i in range(7):
        score = scores[i]
        merged_scores = np.array([np.mean(score[i*split_batch_size:(i+1)*split_batch_size]) for i in range(args.split)])
        print("blur{},{:.2f},{:.2f}".format(i, merged_scores.mean(), merged_scores.std()))
