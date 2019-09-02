# This code is based on https://github.com/sbarratt/inception-score-pytorch
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help='Required. Path to dataset image files')
    parser.add_argument("--model", type=str, required=False, default="/opt/project/CUB_inception_model",
                        help="Required. Path to pre-trained inception model.")
    parser.add_argument("--gpu", type=int, required=False, default=-1,
                        help="Optional. GPU id to use. If not specified, no GPU will be used (CPU only).")
    parser.add_argument("--bs", type=int, required=False, default=32,
                        help="batch size")
    parser.add_argument("--split", type=int, required=False, default=10,
                        help="Optional. Default 10. The number the test set will be split into when aggregation.")
    return parser.parse_args()


def inception_score(imgs, model, device, batch_size=32, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    print("Data num", N)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    # for CUB dataset
    class_num = 200
    inception_model.AuxLogits.fc = nn.Linear(768, class_num)
    inception_model.fc = nn.Linear(2048, class_num)
    inception_model.load_state_dict(torch.load(model))
    for param in inception_model.parameters():
        param.requires_grad = False
    inception_model = inception_model.to(device)
    inception_model.eval()

    # Get predictions
    preds = np.zeros((N, class_num))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batch_size_i = batch.size()[0]

        output = inception_model(batch)
        pred = F.softmax(output, dim=1).data.cpu().numpy()

        preds[i*batch_size:i*batch_size + batch_size_i] = pred

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    args = parse_args()

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    dataset = dset.ImageFolder(
        args.image,
        transform=transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    )
    IgnoreLabelDataset(dataset)

    if args.gpu != -1:
        current_device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        current_device = torch.device("cpu")

    print("Calculating Inception Score...")
    print(inception_score(
        IgnoreLabelDataset(dataset), args.model, current_device, batch_size=args.bs, splits=args.split))
