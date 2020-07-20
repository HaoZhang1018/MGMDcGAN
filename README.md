# MGMDcGAN
This is the code of the following paper (tensorflow):<br>
```
@article{huang2020mgmdcgan,
  title={MGMDcGAN: Medical Image Fusion Using Multi-Generator Multi-Discriminator Conditional Generative Adversarial Network},
  author={Huang, Jun and Le, Zhuliang and Ma, Yong and Fan, Fan and Zhang, Hao and Yang, Lei},
  journal={IEEE Access},
  volume={8},
  pages={55145--55157},
  year={2020},
  publisher={IEEE}
}
```
It is a unified model for multiple image fusion tasks, including:<br>
MRI-PET medical image fusion<br>
MRI-SPECT medical image fusion<br>
CT-SPECT medical image fusion<br>

## Framework:<br>
 Overall procedure:<br>
<div align=center><img src="  " width="440" height="290"/></div><br>


## Fused results:<br>
<div align=center><img src="   " width="900" height="490"/></div>
<div align=center><img src="   " width="900" height="400"/></div>

## To train:<br>
CUDA_VISIBLE_DEVICES=0 python main_medical.py(IS_TRAINING=True)<br>

## To test:<br>
CUDA_VISIBLE_DEVICES=0 python main_medical.py(IS_TRAINING=False)<br>

If you have any question, please email to me (lezhuliang@whu.edu.cn).
