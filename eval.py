from utils import CausalMetric
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, inception_v3, googlenet, vgg16, mobilenet_v2
from saliency.saliency_zoo import big, mfaba_cos, mfaba_norm, mfaba_sharp, mfaba_smooth, agi, ig, sm, sg,deeplift,big_ssa
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3',
                    choices=["inception_v3", "resnet50", "googlenet", "vgg16", "mobilenet_v2"])
parser.add_argument('--generate_from',type=str,default='inception_v3')
parser.add_argument('--attr_method', type=str, default='mfaba_smooth',
                    choices=['big','big_ssa', 'mfaba_cos',"mfaba_difgsmori_smooth","mfaba_tifgsmori_smooth","mfaba_mifgsmori_smooth", 'mfaba_norm', 'mfaba_sharp',"mfaba_ssa_smooth","mfaba_pgd_smooth","mfaba_difgsm_smooth","mfaba_tifgsm_smooth","mfaba_mifgsm_smooth","mfaba_sinifgsm_smooth","mfaba_naa_smooth", 'mfaba_smooth', 'agi', 'ig',  'sm', 'sg','deeplift',"fast_ig","guided_ig","eg","saliencymap"])

args = parser.parse_args()
if args.generate_from == "resnet50" and args.attr_method == "deeplift":
    from resnet_modified import resnet50
    model = resnet50(pretrained=True).eval().to(device)
else:
    model = eval(f"{args.model}(pretrained=True).eval().to(device)")
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
norm_layer = transforms.Normalize(mean, std)
sm = nn.Softmax(dim=-1)
model = nn.Sequential(norm_layer, model, sm).to(device)

deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
insertion = CausalMetric(model, 'ins', 224, substrate_fn=torch.zeros_like)

if __name__ == "__main__":
    attribution = np.load(f"attributions/{args.generate_from}_{args.attr_method}_attributions.npy")
    scores = {'del': deletion.evaluate(
        img_batch, attribution, 100), 'ins': insertion.evaluate(img_batch, attribution, 100)}
    scores['ins'] = np.array(scores['ins'])
    scores['del'] = np.array(scores['del'])
    np.savez(f"scores/{args.model}_{args.generate_from}_{args.attr_method}_scores.npz", **scores)
