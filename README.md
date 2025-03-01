# Unveiling the Unknown: Unleashing the Power of Unknown to Known in Open-Set Source-Free Domain Adaptation


Pytorch code for [Unveiling the Unknown: Unleashing the Power of Unknown to Known in Open-Set Source-Free Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2024/papers/Wan_Unveiling_the_Unknown_Unleashing_the_Power_of_Unknown_to_Known_CVPR_2024_paper.pdf) (CVPR2024)

If you have any questions about our work, please contact me at email wfl128.xidian@gmail.com.

Introduction
---


Specifically, given a source pre-trained model, we first introduce an unknown diffuser that can determine whether classes in space need to be split and merged through similarity measures, to estimate and generate a wider class space distribution, including known and unknown classes. Based on
such a wider space distribution, we enhance the reliability of known class knowledge in the source pre-trained model through contrastive constraint. Finally, various supervision information, including reliable known class knowledge and clustered pseudo-labels, optimize the model for impressive knowledge transfer and generalization. Extensive experiments show that our network can achieve superior exploration and knowledge generalization on unknown classes, while with excellent known class transfer. 

Framework
---
![image-20250301171723334](C:\Users\wan\AppData\Roaming\Typora\typora-user-images\image-20250301171723334.png)



Environment
---
python,numpy, scipy, sklearn,etc.

We have presented the environment file in ./environment.yml.


Dataset
---

We conduct extensive experiments on three standard domain adaptation benchmarks:

- Office
- OfficeHome
- VisDA

Please manually download these datasets from the official websites and put them on the correct position. 



Training
---
