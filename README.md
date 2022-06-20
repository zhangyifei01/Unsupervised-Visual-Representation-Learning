# Unsupervised-Visual-Representation-Learning 
Unsupervised visual representation learning (UVRL) aims at learning generic representations for the initialization of downstream tasks. 
As stated in MoCo, self-supervised learning is a form of unsupervised learning and their distinction is informal in the existing literature. Therefore, it is more inclined to be called UVRL here. So, which one do you prefer?

We list related papers from conferences and journals such as **CVPR**, **ICCV**, **ECCV**, **ICLR**, **ICML**, **NeurIPS**, **AAAI**, **TPAMI**, **TIP**, **TNNLS**, **TCSVT**, **TMM** etc.

Note that only image-level representation learning methods are listed in this repository. Video-level self-supervision and multi-modal self-supervision are to be sorted out.

**Key words:** Unsupervised learning, self-supervised learning, representation learning, pre-training, transfer learning, contrastive learning, pretext task

**关键词：** 无监督学习，自监督学习，表示学习，预训练，迁移学习，对比学习，借口（代理）任务


## 2022
- Masked Autoencoders Are Scalable Vision Learners (**MAE** - <font color="#dd0000">**CVPR**22</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) [[code]](https://github.com/facebookresearch/mae) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/He_Masked_Autoencoders_Are_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research (FAIR)\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Transformer

- SimMIM: A Simple Framework for Masked Image Modeling (**SimMIM** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf) [[code]](https://github.com/microsoft/SimMIM) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Xie_SimMIM_A_Simple_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu \
    > <font color=Gray>**Organization(s)**:</font>  Microsoft Research Asia\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Transformer

- Crafting Better Contrastive Views for Siamese Representation Learning (**ContrastiveCrop** - <font color="#dd0000">**CVPR**22</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Crafting_Better_Contrastive_Views_for_Siamese_Representation_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/xyupeng/ContrastiveCrop)
    > <font color=Gray>**Author(s)**:</font> Xiangyu Peng, Kai Wang, Zheng Zhu, Mang Wang, Yang You\
    > <font color=Gray>**Organization(s)**:</font> National University of Singapore; Tsinghua University; Alibaba Group\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- A Simple Data Mixing Prior for Improving Self-Supervised Learning (**SDMP** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_A_Simple_Data_Mixing_Prior_for_Improving_Self-Supervised_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/OliverRensu/SDMP)
    > <font color=Gray>**Author(s)**:</font> Sucheng Ren, Huiyu Wang, Zhengqi Gao, Shengfeng He, Alan Yuille, Yuyin Zhou, Cihang Xie\
    > <font color=Gray>**Organization(s)**:</font> South China University of Technology; Johns Hopkins University; Massachusetts Institute of Technology; UC Santa Cruz\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- On the Importance of Asymmetry for Siamese Representation Learning (**asym-siam** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_On_the_Importance_of_Asymmetry_for_Siamese_Representation_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/facebookresearch/asym-siam) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_On_the_Importance_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, Xinlei Chen\
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research (FAIR); Purdue University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- HCSC: Hierarchical Contrastive Selective Coding (**HCSC** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_HCSC_Hierarchical_Contrastive_Selective_Coding_CVPR_2022_paper.pdf) [[code]](https://github.com/gyfastas/HCSC) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Guo_HCSC_Hierarchical_Contrastive_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yuanfan Guo, Minghao Xu, Jiawen Li, Bingbing Ni, Xuanyu Zhu, Zhenbang Sun, Yi Xu \
    > <font color=Gray>**Organization(s)**:</font> MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University; 2Mila - Qu´ebec AI Institute, University of Montr´eal; ByteDance\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Unsupervised Representation Learning for Binary Networks by Joint Classifier Learning (**BURN** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Unsupervised_Representation_Learning_for_Binary_Networks_by_Joint_Classifier_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/naver-ai/burn) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Kim_Unsupervised_Representation_Learning_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Dahyun Kim, Jonghyun Choi \
    > <font color=Gray>**Organization(s)**:</font> Upstage AI Research; NAVER AI Lab.; Yonsei University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Unleashing Potential of Unsupervised Pre-Training with Intra-Identity Regularization for Person Re-Identification (**UP-ReID** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Unleashing_Potential_of_Unsupervised_Pre-Training_With_Intra-Identity_Regularization_for_Person_CVPR_2022_paper.pdf) [[code]](https://github.com/Frost-Yang-99/UP-ReID) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Yang_Unleashing_Potential_of_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Zizheng Yang, Xin Jin, Kecheng Zheng, Feng Zhao \
    > <font color=Gray>**Organization(s)**:</font> University of Science and Technology of China\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- UniVIP: A Unified Framework for Self-Supervised Visual Pre-training (**UniVIP** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_UniVIP_A_Unified_Framework_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font> Zhaowen Li, Yousong Zhu, Fan Yang, Wei Li, Chaoyang Zhao, Yingying Chen, Zhiyang Chen, Jiahao Xie, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang \
    > <font color=Gray>**Organization(s)**:</font> National Laboratory of Pattern Recognition, Institute of Automation, CAS; School of Artificial Intelligence, University of Chinese Academy of Sciences; SenseTime Research; etc.\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Semantic-Aware Auto-Encoders for Self-supervised Representation Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Semantic-Aware_Auto-Encoders_for_Self-Supervised_Representation_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/wanggrun/Semantic-Aware-AE)
    > <font color=Gray>**Author(s)**:</font> Guangrun Wang, Yansong Tang, Liang Lin, Philip H.S. Torr \
    > <font color=Gray>**Organization(s)**:</font> University of Oxford; Tsinghua-Berkeley Shenzhen Institute, Tsinghua University; Sun Yat-sen University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Rethinking the Augmentation Module in Contrastive Learning: Learning Hierarchical Augmentation Invariance with Expanded Views (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font> Junbo Zhang, Kaisheng Ma \
    > <font color=Gray>**Organization(s)**:</font> Tsinghua University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy (**LoGo** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Leverage_Your_Local_and_Global_Representations_A_New_Self-Supervised_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/ztt1024/LoGo-SSL) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_Leverage_Your_Local_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Tong Zhang, Congpei Qiu, Wei Ke, Sabine Süsstrunk, Mathieu Salzmann \
    > <font color=Gray>**Organization(s)**:</font> School of Computer and Communication Sciences, EPFL; Xi’an Jiaotong University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Learning Where to Learn in Cross-View Self-Supervised Learning (**LEWEL** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learning_Where_To_Learn_in_Cross-View_Self-Supervised_Learning_CVPR_2022_paper.pdf) [[code]](https://t.ly/ZI0A) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Huang_Learning_Where_To_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Lang Huang, Shan You, Mingkai Zheng, Fei Wang, Chen Qian, Toshihiko Yamasaki \
    > <font color=Gray>**Organization(s)**:</font> The University of Tokyo; SenseTime Research; The University of Sydney\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Exploring the Equivalence of Siamese Self-Supervised Learning via A Unified Gradient Framework (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_Exploring_the_Equivalence_of_Siamese_Self-Supervised_Learning_via_a_Unified_CVPR_2022_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Tao_Exploring_the_Equivalence_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Chenxin Tao, Honghui Wang, Xizhou Zhu, Jiahua Dong, Shiji Song, Gao Huang, Jifeng Dai \
    > <font color=Gray>**Organization(s)**:</font> Tsinghua University; SenseTime Research; Zhejiang University; Beijing Academy of Artificial Intelligence\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Dual Temperature Helps Contrastive Learning Without Many Negative Samples: Towards Understanding and Simplifying MoCo (**DT** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Dual_Temperature_Helps_Contrastive_Learning_Without_Many_Negative_Samples_Towards_CVPR_2022_paper.pdf) [[code]](https://bit.ly/3LkQbaT) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_Dual_Temperature_Helps_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Chaoning Zhang, Kang Zhang, Trung X. Pham, Axi Niu, Zhinan Qiao, Chang D. Yoo, In So Kweon \
    > <font color=Gray>**Organization(s)**:</font>  KAIST; Northwestern Polytechnical University; University of North Texas\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Directional Self-supervised Learning for Heavy Image Augmentations (**DSSL** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Bai_Directional_Self-Supervised_Learning_for_Heavy_Image_Augmentations_CVPR_2022_paper.pdf) [[code]](https://github.com/Yif-Yang/DSSL) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Bai_Directional_Self-Supervised_Learning_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yalong Bai, Yifan Yang, Wei Zhang, Tao Mei \
    > <font color=Gray>**Organization(s)**:</font>  JD AI Research; Peking University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Unsupervised Visual Representation Learning by Online Constrained K-Means (**CoKe** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Qian_Unsupervised_Visual_Representation_Learning_by_Online_Constrained_K-Means_CVPR_2022_paper.pdf) [[code]](https://github.com/idstcv/CoKe) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Qian_Unsupervised_Visual_Representation_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Qi Qian, Yuanhong Xu, Juhua Hu, Hao Li, Rong Jin \
    > <font color=Gray>**Organization(s)**:</font>  Alibaba Group, Bellevue; Alibaba Group, Hangzhou; University of Washington\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Align Representations with Base: A New Approach to Self-Supervised Learning (**ARB** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Align_Representations_With_Base_A_New_Approach_to_Self-Supervised_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/Sherrylone/Align-Representation-with-Base)
    > <font color=Gray>**Author(s)**:</font> Shaofeng Zhang, Lyn Qiu, Feng Zhu, Junchi Yan, Hengrui Zhang, Rui Zhao, Hongyang Li, Xiaokang Yang \
    > <font color=Gray>**Organization(s)**:</font>  Shanghai Jiao Tong University; SenseTime Research\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Exploring Set Similarity for Dense Self-supervised Representation Learning (**SetSim** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Align_Representations_With_Base_A_New_Approach_to_Self-Supervised_Learning_CVPR_2022_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font> Shaofeng Zhang, Lyn Qiu, Feng Zhu, Junchi Yan, Hengrui Zhang, Rui Zhao, Hongyang Li, Xiaokang Yang \
    > <font color=Gray>**Organization(s)**:</font>  University of Sydney; Kuaishou Technology; Xidian University; University of Melbourne\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Dense representation

- Patch-level Representation Learning for Self-supervised Vision Transformers (**SelfPatch** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yun_Patch-Level_Representation_Learning_for_Self-Supervised_Vision_Transformers_CVPR_2022_paper.pdf) [[code]](https://github.com/alinlab/SelfPatch) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Yun_Patch-Level_Representation_Learning_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Sukmin Yun, Hankook Lee, Jaehyung Kim, Jinwoo Shin \
    > <font color=Gray>**Organization(s)**:</font>  Korea Advanced Institute of Science and Technology (KAIST)\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Dense representation

- Self-Supervised Learning of Object Parts for Semantic Segmentation (**Leopart** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf) [[No code]]( https://github.com/MkuuWaUjinga/leopart) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ziegler_Self-Supervised_Learning_of_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Adrian Ziegler, Yuki M. Asano \
    > <font color=Gray>**Organization(s)**:</font>  Technical University of Munich; QUVA Lab, University of Amsterdam\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Dense representation

- Point-Level Region Contrast for Object Detection Pre-Training (**** - <font color="#dd0000">**CVPR**22</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Bai_Point-Level_Region_Contrast_for_Object_Detection_Pre-Training_CVPR_2022_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Bai_Point-Level_Region_Contrast_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yutong Bai, Xinlei Chen, Alexander Kirillov, Alan Yuille, Alexander C. Berg \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research (FAIR); Johns Hopkins University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Dense representation

- Use All The Labels: A Hierarchical Multi-Label Contrastive Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Use_All_the_Labels_A_Hierarchical_Multi-Label_Contrastive_Learning_Framework_CVPR_2022_paper.pdf) [[code]](https://github.com/salesforce/hierarchicalContrastiveLearning) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_Use_All_the_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Shu Zhang, Ran Xu, Caiming Xiong, Chetan Ramaiah \
    > <font color=Gray>**Organization(s)**:</font>  Salesforce Research\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Label supervision
    
- Revisiting the Transferability of Supervised Pretraining: an MLP Perspective (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Revisiting_the_Transferability_of_Supervised_Pretraining_An_MLP_Perspective_CVPR_2022_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Revisiting_the_Transferability_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yizhou Wang, Shixiang Tang, Feng Zhu, Lei Bai, Rui Zhao, Donglian Qi, Wanli Ouyang \
    > <font color=Gray>**Organization(s)**:</font>  Zhejiang University; The University of Sydney; SenseTime Research; Shanghai Jiao Tong University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Label supervision

- Does Robustness on ImageNet Transfer to Downstream Tasks? (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yamada_Does_Robustness_on_ImageNet_Transfer_to_Downstream_Tasks_CVPR_2022_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Yamada_Does_Robustness_on_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yutaro Yamada, Mayu Otani \
    > <font color=Gray>**Organization(s)**:</font>  Yale University; CyberAgent, Inc. \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- When Does Contrastive Visual Representation Learning Work? (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cole_When_Does_Contrastive_Visual_Representation_Learning_Work_CVPR_2022_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font> Elijah Cole, Xuan Yang, Kimberly Wilber, Oisin Mac Aodha, Serge Belongie \
    > <font color=Gray>**Organization(s)**:</font>  Caltech; Google; University of Edinburgh; Alan Turing Institute; University of Copenhagen\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- Rethinking Minimal Sufficient Representation in Contrastive Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Rethinking_Minimal_Sufficient_Representation_in_Contrastive_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/Haoqing-Wang/InfoCL) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Rethinking_Minimal_Sufficient_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Haoqing Wang, Xun Guo, Zhi-Hong Deng, Yan Lu \
    > <font color=Gray>**Organization(s)**:</font>  Peking University; Microsoft Research Asia\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- Beyond Supervised vs. Unsupervised: Representative Benchmarking and Analysis of Image Representation Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gwilliam_Beyond_Supervised_vs._Unsupervised_Representative_Benchmarking_and_Analysis_of_Image_CVPR_2022_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Gwilliam_Beyond_Supervised_vs._CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Matthew Gwilliam, Abhinav Shrivastava \
    > <font color=Gray>**Organization(s)**:</font>  University of Maryland, College Park\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- DETReg: Unsupervised Pretraining with
Region Priors for Object Detection (**DETReg** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Bar_DETReg_Unsupervised_Pretraining_With_Region_Priors_for_Object_Detection_CVPR_2022_paper.pdf) [[code]](https://www.amirbar.net/detreg/) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Bar_DETReg_Unsupervised_Pretraining_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Amir Bar, Xin Wang, Vadim Kantorov, Colorado J. Reed, Roei Herzig, Gal Chechik, Anna Rohrbach, Trevor Darrell, Amir Globerson \
    > <font color=Gray>**Organization(s)**:</font>  Tel-Aviv University; Berkeley AI Research; NVIDIA;  Bar-Ilan University; Microsoft Research\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Transformer

- Masked Feature Prediction for Self-Supervised Visual Pre-Training (**MaskFeat** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wei_Masked_Feature_Prediction_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, Christoph Feichtenhofer \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research; Johns Hopkins University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Transformer

- CoDo: Contrastive Learning with Downstream Background Invariance for Detection (**CoDo** - <font color="#dd0000">**CVPRW**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Zhao_CoDo_Contrastive_Learning_With_Downstream_Background_Invariance_for_Detection_CVPRW_2022_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font> Bing Zhao, Jun Li, Hong Zhu \
    > <font color=Gray>**Organization(s)**:</font>  Department of AI and HPC Inspur Electronic Information Industry Co., Ltd\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Dense representation

- What Should Be Equivariant In Self-Supervised Learning (**** - <font color="#dd0000">**CVPRW**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Xie_What_Should_Be_Equivariant_in_Self-Supervised_Learning_CVPRW_2022_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font> Yuyang Xie, Jianhong Wen, Kin Wai Lau, Yasar Abbas Ur Rehman, Jiajun Shen \
    > <font color=Gray>**Organization(s)**:</font>  TCL AI Lab; Fuzhou University; City University of Hong Kong\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- Where are my Neighbors? Exploiting Patches Relations in Self-Supervised Vision Transformer (**** - <font color="#dd0000">**CVPRW**22</font>) [[paper]](https://arxiv.org/abs/2206.00481) [[code]](https://github.com/guglielmocamporese/relvit)
    > <font color=Gray>**Author(s)**:</font> Guglielmo Camporese, Elena Izzo, Lamberto Ballan \
    > <font color=Gray>**Organization(s)**:</font>  University of Padova, Italy\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

## 2021

- Exploring Simple Siamese Representation Learning (**SimSiam** - <font color="#dd0000">**CVPR**21</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/facebookresearch/simsiam) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Chen_Exploring_Simple_Siamese_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Xinlei Chen, Kaiming He \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research (FAIR)\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Siamese architecture

- Jigsaw Clustering for Unsupervised Visual Representation Learning (**JigClu** - <font color="#dd0000">**CVPR**21</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Jigsaw_Clustering_for_Unsupervised_Visual_Representation_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/Jia-Research-Lab/JigsawClustering) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Chen_Jigsaw_Clustering_for_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Pengguang Chen, Shu Liu, Jiaya Jia \
    > <font color=Gray>**Organization(s)**:</font>  The Chinese University of Hong Kong; SmartMore\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Single branch

- OBoW: Online Bag-of-Visual-Words Generation for Self-Supervised Learning (**OBoW** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Gidaris_OBoW_Online_Bag-of-Visual-Words_Generation_for_Self-Supervised_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/valeoai/obow) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Gidaris_OBoW_Online_Bag-of-Visual-Words_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Spyros Gidaris, Andrei Bursuc, Gilles Puy, Nikos Komodakis, Matthieu Cord, Patrick Perez \
    > <font color=Gray>**Organization(s)**:</font>  valeo.ai; University of Crete; Sorbonne Université\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Spatial Assembly Networks for Image Representation Learning (**SAN** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spatial_Assembly_Networks_for_Image_Representation_Learning_CVPR_2021_paper.pdf) [[No code]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Spatial_Assembly_Networks_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  University of Missouri, MO; Beijing Jiaotong University; Shenzhen University\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- SelfAugment: Automatic Augmentation Policies for Self-Supervised Learning (**SelfAugment** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Reed_SelfAugment_Automatic_Augmentation_Policies_for_Self-Supervised_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/cjrd/selfaugment) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Reed_SelfAugment_Automatic_Augmentation_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  BAIR, Department of Computer Science, UC Berkeley; Graduate Group in Bioengineering (Berkeley/UCSF), Weill Neurosciences Institute & UCSF Neurological Surgery \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Progressive Stage-wise Learning for Unsupervised Feature Representation Enhancement (**PSL** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Stage-Wise_Learning_for_Unsupervised_Feature_Representation_Enhancement_CVPR_2021_paper.pdf) [[No code]]()
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  Shanghai Jiao Tong University; Johns Hopkins University; Peking University; MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning (**PixPro** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Propagate_Yourself_Exploring_Pixel-Level_Consistency_for_Unsupervised_Visual_Representation_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/zdaxie/PixPro)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  Tsinghua University; Xi’an Jiaotong University; Microsoft Research Asia \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Dense representation

- How Well Do Self-Supervised Models Transfer? (**** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ericsson_How_Well_Do_Self-Supervised_Models_Transfer_CVPR_2021_paper.pdf) [[code]](https://github.com/linusericsson/ssl-transfer) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Ericsson_How_Well_Do_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  University of Edinburgh; Samsung AI Research, Cambridge \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Analysis

- Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning (**EMAN** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Exponential_Moving_Average_Normalization_for_Self-Supervised_and_Semi-Supervised_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/amazon-research/exponential-moving-average-normalization) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Cai_Exponential_Moving_Average_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  Amazon Web Services \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Dense Contrastive Learning for Self-Supervised Visual Pre-Training (**DenseCL** - <font color="#dd0000">**CVPR**21</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Dense_Contrastive_Learning_for_Self-Supervised_Visual_Pre-Training_CVPR_2021_paper.pdf) [[code]](https://github.com/WXinlong/DenseCL) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Dense_Contrastive_Learning_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  The University of Adelaide; Tongji University; ByteDance AI Lab \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Dense representation

- Unsupervised Feature Learning by Cross-Level Instance-Group Discrimination (**CLD** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Feature_Learning_by_Cross-Level_Instance-Group_Discrimination_CVPR_2021_paper.pdf) [[code]]( https://github.com/frankxwang/CLD-UnsupervisedLearning) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Unsupervised_Feature_Learning_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  UC Berkeley / ICSI; S-Lab, NTU \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries (**AdCo** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_AdCo_Adversarial_Contrast_for_Efficient_Learning_of_Unsupervised_Representations_From_CVPR_2021_paper.pdf) [[code]](https://github.com/maple-research-lab/AdCo) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Hu_AdCo_Adversarial_Contrast_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font>  \
    > <font color=Gray>**Organization(s)**:</font>  Peking University; Purdue University; Laboratory for MAchine Perception and LEarning (MAPLE) \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

## 2020

- Momentum Contrast for Unsupervised Visual Representation Learning (**MoCo** - <font color="#dd0000">**CVPR**20</font> Oral) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) [[code]](https://github.com/facebookresearch/moco)
    > <font color=Gray>**Author(s)**:</font> Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research (FAIR)\
    > <font color=Gray>**Description**:</font>  Building a dynamic dictionary with a queue and a moving-averaged encoder \
    > <font color=Gray>**Tags**:</font> InfoNCE loss, siamese architecture


## 2019

- Unsupervised Embedding Learning via Invariant and Spreading Instance Feature (**ISIF** - <font color="#dd0000">**CVPR**19</font>) [[paper]]() [[code]]()
    > <font color=Gray>**Author(s)**:</font> Mang Ye, Xu Zhang,  Pong C. Yuen, Shih-Fu Chang\
    > <font color=Gray>**Organization(s)**:</font>  Hong Kong Baptist University; Columbia University\
    > <font color=Gray>**Description**:</font>  Learning augmentation invariant  and instance spread-out features \
    > <font color=Gray>**Tags**:</font> Siamese architecture

## 2018

- Unsupervised Feature Learning via Non-Parametric Instance Discrimination ( **InsDis** - <font color="#dd0000">**CVPR**18</font> Spotlight) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf) [[code]](https://github.com/zhirongw/lemniscate.pytorch)
    > <font color=Gray>**Author(s)**:</font> Zhirong Wu, Yuanjun Xiong, Stella X. Yu, Dahua Lin \
    > <font color=Gray>**Organization(s)**:</font>  UC Berkeley/ICSI; Chinese University of Hong Kong; Amazon Rekognition\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Contrastive learning, single branch


## 2017

- Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction (**Split-Brain Auto** - <font color="#dd0000">**CVPR**17</font> ) [[paper]]() [[code]]()
    > <font color=Gray>**Author(s)**:</font> Richard Zhang, Phillip Isola, Alexei A. Efros\
    > <font color=Gray>**Organization(s)**:</font>  Berkeley AI Research (BAIR) Laboratory, University of California, Berkeley\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

## 2016

- Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles (**Jigsaw** - <font color="#dd0000">**ECCV**16</font> ) [[paper]]() [[code]]()
    > <font color=Gray>**Author(s)**:</font> Mehdi Noroozi, Paolo Favaro\
    > <font color=Gray>**Organization(s)**:</font>  Carnegie Mellon University; University of California, Berkeley\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

## 2015

- Unsupervised Visual Representation Learning by Context Prediction (**Context** - <font color="#dd0000">**ICCV**15</font> ) [[paper]]() [[code]]()
    > <font color=Gray>**Author(s)**:</font> Carl Doersch, Abhinav Gupta, Alexei A. Efros\
    > <font color=Gray>**Organization(s)**:</font>  Institute for Informatiks, University of Bern\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

## 2014

- Discriminative Unsupervised Feature Learning with Convolutional Neural Networks (**ExemplarCNN** - <font color="#dd0000">**NeurIPS**14</font> ) [[paper]]() [[code]]()
    > <font color=Gray>**Author(s)**:</font> Alexey Dosovitskiy, Jost Tobias Springenberg, Martin A. Riedmiller, Thomas Brox\
    > <font color=Gray>**Organization(s)**:</font>  Computer Science Department, University of Freiburg\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task, instance discrimination


***

Thanks for the support of Prof. [Yu Zhou](https://people.ucas.ac.cn/~yuzhou).
