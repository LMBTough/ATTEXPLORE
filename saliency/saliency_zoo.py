from saliency.core import pgd_step,pgd_ssa_step,DL, BIG,SSA, FGSM, AMPE, IntegratedGradient, SaliencyGradient, SmoothGradient,FGSMGradSSA
from saliency.core import FastIG, GuidedIG,SaliencyMap,AttributionPriorExplainer
import torch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import torch.nn.functional as F

import cv2
import numpy as np
import torch

def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
        
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result




class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model,model_name):
        self.model_arch = model
        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        if model_name == 'resnet50':
            target_layer = model[1]._modules.get('layer3')
        elif model_name == 'vgg16':
            target_layer = model[1]._modules.get('features')[29]
        elif model_name == 'inception_v3':
            target_layer = model[1]._modules.get('Mixed_7c')

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map.cpu().detach().numpy(), logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
    

def gradcam(model_name,model,data,target):
    gd = GradCAM(model,model_name)
    saliency_map, logit = gd(data)
    return saliency_map


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20):
    model = model[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def agi_ssa(model, data, target, epsilon=0.05, max_iter=20, topk=1):
    model = model[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_ssa_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def big(model, data, target, data_min=0, data_max=1, epsilons=[16], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map

def big_ssa(model, data, target, data_min=0, data_max=1, epsilons=[16], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [SSA(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map



def ampe(model, data, target, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = AMPE(model)
    epsilon = epsilon / 255
    attack = FGSMGradSSA(
        epsilon=epsilon, data_min=data_min, data_max=data_max,N=N)
    _, _, _, hats, grads = attack(
        model, data, target,num_steps=num_steps, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


def ig(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)


def sg(model, data, target, stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model, stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)


def deeplift(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)

import copy
def fast_ig(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)

def guided_ig(model, data, target, steps=15):
    model = copy.deepcopy(model)[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        # m = torch.nn.Softmax(dim=1)
        # output = m(output)
        output = output
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    result = method.GetMask(
        im, call_model_function, call_model_args, x_steps=steps, x_baseline=baseline)
    return np.expand_dims(result, axis=0)

def saliencymap(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = SaliencyMap(model)
    return saliencymap(data, target)


def eg(model,dataloader,data,target,*args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    APExp = AttributionPriorExplainer(dataloader.dataset, 4,k=1)
    attr_eg = APExp.shap_values(model,data).cpu().detach().numpy()
    return attr_eg