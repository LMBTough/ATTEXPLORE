<div align="center">

# AttEXplore: Attribution for Explanation with model parameters eXploration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue:ICRL 2023](https://img.shields.io/badge/Venue-ICRL%202024-007CFF)](https://openreview.net/forum?id=FsVxd9CIlb)

</div>

# Abstract

AttEXplore introduces a novel method for explaining deep neural network decisions by exploring model parameters. This approach leverages the concepts from transferable adversarial attacks to identify the most influential features affecting model decisions, offering a unique blend of robustness and interpretability. Our method not only outperforms traditional attribution methods across various benchmarks but also provides deeper insights into the decision-making processes of complex models. By integrating model parameter exploration, AttEXplore enhances the transparency of model predictions, making it a valuable tool for applications requiring high levels of trust and accountability. [[Paper Link]](https://openreview.net/forum?id=FsVxd9CIlb) [[Slide Link]](https://iclr.cc/media/iclr-2024/Slides/19046.pdf)


# Experiments

To run the code, you need to install the following packages use environment.yml:
```
conda env create -f environment.yml
```


pretrained models are available at torchvision

#### Introduction


- `AMPE/core/ampe.py` : the code for AttEXplore.

- `eval.py` : the code for deletion/insertion metric.
  

#### Example Usage

##### Generate adversarial examples:

- AttEXplore

```
python generate_attributions.py --attr_method ampe --model inception_v3
```

You can also modify the hyper parameter values to align with the detailed setting in our paper.


##### Deletion/Insertion metric:

```
python eval.py --attr_method ampe --model inception_v3 --generate_from inception_v3
```

## Citing AttEXplore
If you utilize this implementation or the AttEXplore methodology in your research, please cite the following paper:

```
@inproceedings{zhu2023attexplore,
  title={AttEXplore: Attribution for Explanation with model parameters eXploration},
  author={Zhu, Zhiyu and Chen, Huaming and Zhang, Jiayu and Wang, Xinyi and Jin, Zhibo and Xue, Jason and Salim, Flora D},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```