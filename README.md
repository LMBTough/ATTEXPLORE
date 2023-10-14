# AMPE
This repository contains the code for the paper "AMPE"

To run the code, you need to install the following packages use environment.yml:
```
conda env create -f environment.yml
```

# Experiments

pretrained models are available at torchvision

#### Introduction


- `AMPE/core/ampe.py` : the code for AMPE.

- `eval.py` : the code for deletion/insertion metric.
  

#### Example Usage

##### Generate adversarial examples:

- AMPE

```
python generate_attributions.py --attr_method ampe --model inception_v3
```

You can also modify the hyper parameter values to align with the detailed setting in our paper.


##### Deletion/Insertion metric:

```
python eval.py --attr_method ampe --model inception_v3 --generate_from inception_v3
```