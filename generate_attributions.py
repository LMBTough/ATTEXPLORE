import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, inception_v3, googlenet, vgg16, mobilenet_v2
from saliency.saliency_zoo import big,big_ssa, mfaba_cos, mfaba_norm, mfaba_sharp,mfaba_pgd_smooth,mfaba_difgsm_smooth,mfaba_tifgsm_smooth,mfaba_mifgsm_smooth,mfaba_sinifgsm_smooth,mfaba_naa_smooth, mfaba_smooth, agi, ig, sm, sg,deeplift,mfaba_ssa_smooth
from saliency.saliency_zoo import mfaba_difgsmori_smooth,mfaba_tifgsmori_smooth,mfaba_mifgsmori_smooth
from saliency.saliency_zoo import fast_ig,guided_ig,eg,saliencymap
from tqdm import tqdm
import torch
import numpy as np
import argparse
import torch
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_batch = torch.load("data/img_batch.pt")
target_batch = torch.load("data/label_batch.pt")
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(img_batch, target_batch)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3',
                    choices=["inception_v3", "resnet50", "googlenet", "vgg16", "mobilenet_v2"])
parser.add_argument('--attr_method', type=str, default='mfaba_smooth',
                    choices=['big','big_ssa', 'mfaba_cos',"mfaba_difgsmori_smooth","mfaba_tifgsmori_smooth","mfaba_mifgsmori_smooth", 'mfaba_norm', 'mfaba_sharp',"mfaba_ssa_smooth","mfaba_pgd_smooth","mfaba_difgsm_smooth","mfaba_tifgsm_smooth","mfaba_mifgsm_smooth","mfaba_sinifgsm_smooth","mfaba_naa_smooth", 'mfaba_smooth', 'agi', 'ig',  'sm', 'sg','deeplift',"fast_ig","guided_ig","eg","saliencymap"])

args = parser.parse_args()

attr_method = eval(args.attr_method)

if args.model == "resnet50" and args.attr_method == "deeplift":
    from resnet_modified import resnet50
    model = resnet50(pretrained=True).eval().to(device)
else:
    model = eval(f"{args.model}(pretrained=True).eval().to(device)")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = transforms.Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model,sm).to(device)

if __name__ == "__main__":
    attributions = []
    if args.attr_method == 'mfaba_cos' or args.attr_method == 'mfaba_norm':
        batch_size = 1
    elif args.attr_method == "mfaba_naa_smooth":
        batch_size = 2
    elif args.attr_method == 'big':
        batch_size = 4
    elif args.attr_method == 'big_ssa':
        batch_size = 4
    elif args.attr_method == 'agi':
        batch_size = 64
    elif args.attr_method == 'ig':
        batch_size = 4
    elif args.attr_method.startswith('mfaba'):
        batch_size = 128
    elif args.attr_method == 'sm':
        batch_size = 64
    elif args.attr_method == 'sg':
        batch_size = 4
    elif args.attr_method == 'deeplift':
        batch_size = 4
    elif args.attr_method == 'guided_ig':
        batch_size = 1
    else:
        batch_size = 4
    import os
    if not os.path.exists("attributions/" + args.model+"_" +
            args.attr_method+"_attributions.npy"):
        for i in tqdm(range(0, len(img_batch), batch_size)):
            img = img_batch[i:i+batch_size].to(device)
            target = target_batch[i:i+batch_size].to(device)
            if args.attr_method == "eg":
                attributions.append(attr_method(model,dataloader, img, target))
            else:
                attributions.append(attr_method(model, img, target))
        if attributions[0].shape.__len__() == 3:
            attributions = [np.expand_dims(attribution, axis=0) for attribution in attributions]
        attributions = np.concatenate(attributions, axis=0)
        np.save("attributions/" + args.model+"_" +
                args.attr_method+"_attributions.npy", attributions)
