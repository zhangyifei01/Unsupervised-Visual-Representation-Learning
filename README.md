# Unsupervised/Self-Supervised-Visual-Representation-Learning 
## 无监督/自监督视觉表示学习

**[ Updating ...... ]**

Unsupervised visual representation learning (UVRL) aims at learning generic representations for the initialization of downstream tasks. 
As stated in MoCo, self-supervised learning is a form of unsupervised learning and their distinction is informal in the existing literature. Therefore, it is more inclined to be called UVRL here. So, which one do you prefer?

We list related papers from conferences and journals such as **[CVPR](https://openaccess.thecvf.com/menu)**, **[ICCV](https://openaccess.thecvf.com/menu)**, **[ECCV](https://www.ecva.net/)**, **[ICLR](https://openreview.net/group?id=ICLR.cc&referrer=%5BHomepage%5D(%2F))**, **[ICML](https://proceedings.mlr.press/)**, **[NeurIPS](https://nips.cc/)**, **[AAAI](https://aaai.org/Library/conferences-library.php)**, **[TPAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34)**, **TIP**, **TNNLS**, **TCSVT**, **TMM** etc.

Note that only image-level representation learning methods are listed in this repository. Video-level self-supervision and multi-modal self-supervision are to be sorted out.

**Key words:** Unsupervised learning, self-supervised learning, representation learning, pre-training, transfer learning, contrastive learning, pretext task

**关键词：** 无监督学习，自监督学习，表示学习，预训练，迁移学习，对比学习，借口（代理）任务


## 2022
- Decoupled Contrastive Learning (**DCL** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2110.06848) 
    > <font color=Gray>**Author(s)**:</font> Chun-Hsiao Yeh, Cheng-Yao Hong, Yen-Chi Hsu, Tyng-Luh Liu, Yubei Chen, Yann LeCun \
    > <font color=Gray>**Organization(s)**:</font>  IIS, Academia Sinica; UC Berkeley; National Taiwan University; Meta AI Research; New York University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Equivariance and Invariance Inductive Bias for
Learning from Insufficient Data (**EqInv** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.12258) [[code]](https://github.com/Wangt-CN/EqInv)
    > <font color=Gray>**Author(s)**:</font> Tan Wang, Qianru Sun, Sugiri Pranata, Karlekar Jayashree, Hanwang Zhang \
    > <font color=Gray>**Organization(s)**:</font>  Nanyang Technological University; Singapore Management University; Panasonic R&D Center Singapore\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Fast-MoCo: Boost Momentum-based Contrastive Learning with Combinatorial Patches (**Fast-MoCo** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.08220) 
    > <font color=Gray>**Author(s)**:</font> Yuanzheng Ci, Chen Lin, Lei Bai, Wanli Ouyang \
    > <font color=Gray>**Organization(s)**:</font>  The University of Sydney; University of Oxford; Shanghai AI Laboratory\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Unsupervised Visual Representation Learning by Synchronous Momentum Grouping (**SMoG** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.06167) 
    > <font color=Gray>**Author(s)**:</font> Bo Pang, Yifan Zhang, Yaoyi Li, Jia Cai, Cewu Lu \
    > <font color=Gray>**Organization(s)**:</font>  Shanghai Jiao Tong University; HuaWei \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervision Can Be a Good Few-Shot Learner (**UniSiam** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.09176) 
    > <font color=Gray>**Author(s)**:</font> Yuning Lu, Liangjian Wen, Jianzhuang Liu, Yajing Liu, Xinmei Tian \
    > <font color=Gray>**Organization(s)**:</font>  University of Science and Technology of China; Huawei Noah’s Ark Lab; Hefei Comprehensive National Science Center \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  few-shot

- ConCL: Concept Contrastive Learning for Dense Prediction Pre-training in Pathology Images (**ConCL** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.06733) 
    > <font color=Gray>**Author(s)**:</font> Jiawei Yang, Hanbo Chen, Yuan Liang, Junzhou Huang, Lei He, Jianhua Yao \
    > <font color=Gray>**Organization(s)**:</font>  Tencent AI Lab; UCLA; University of Texas at Arlington \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  dense, medical

- Dense Siamese Network for Dense Unsupervised Learning (**DenseSiam** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2203.11075) [[code]](https://github.com/ZwwWayne/DenseSiam)
    > <font color=Gray>**Author(s)**:</font> Wenwei Zhang, Jiangmiao Pang, Kai Chen, Chen Change Loy \
    > <font color=Gray>**Organization(s)**:</font> Nanyang Technological University; Shanghai AI Laboratory; SenseTime Research \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  dense

- Bootstrapped Masked Autoencoders for Vision BERT Pretraining (**BootMAE** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.07116) [[code]](https://github.com/LightDXY/BootMAE)
    > <font color=Gray>**Author(s)**:</font> Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu \
    > <font color=Gray>**Organization(s)**:</font> University of Science and Technology of China; Microsoft Research Asia; Microsoft Cloud + AI \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  transformer

- SdAE: Self-distillated Masked Autoencoder (**SdAE** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2208.00449) [[code]](https://github.com/AbrahamYabo/SdAE)
    > <font color=Gray>**Author(s)**:</font> Yabo Chen, Yuchen Liu, Dongsheng Jiang, Xiaopeng Zhang, Wenrui Dai, Hongkai Xiong, Qi Tian \
    > <font color=Gray>**Organization(s)**:</font>  Shanghai Jiao Tong University; Huawei Cloud EI \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  transformer

- Contrastive Vision-Language Pre-training with Limited Resources (**ZeroVL** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2112.09331) [[code]](https://github.com/zerovl/ZeroVL)
    > <font color=Gray>**Author(s)**:</font> Quan Cui, Boyan Zhou, Yu Guo, Weidong Yin, Hao Wu, Osamu Yoshie, Yubo Chen \
    > <font color=Gray>**Organization(s)**:</font>  ByteDance; Waseda University \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  VL

- Tailoring Self-Supervision for Supervised Learning (**LoRot** - <font color="#dd0000">**ECCV**22</font>) [[paper]](https://arxiv.org/abs/2207.10023) [[code]](https://github.com/wjun0830/Localizable-Rotation)
    > <font color=Gray>**Author(s)**:</font> WonJun Moon, Ji-Hwan Kim, Jae-Pil Heo \
    > <font color=Gray>**Organization(s)**:</font>  Sungkyunkwan University \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font>  supervised

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
    > <font color=Gray>**Tags**:</font> Clustering

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

- UniVIP: A Unified Framework for Self-Supervised Visual Pre-training (**UniVIP** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_UniVIP_A_Unified_Framework_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Zhaowen Li, Yousong Zhu, Fan Yang, Wei Li, Chaoyang Zhao, Yingying Chen, Zhiyang Chen, Jiahao Xie, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang \
    > <font color=Gray>**Organization(s)**:</font> National Laboratory of Pattern Recognition, Institute of Automation, CAS; School of Artificial Intelligence, University of Chinese Academy of Sciences; SenseTime Research; etc.\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Semantic-Aware Auto-Encoders for Self-supervised Representation Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Semantic-Aware_Auto-Encoders_for_Self-Supervised_Representation_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/wanggrun/Semantic-Aware-AE)
    > <font color=Gray>**Author(s)**:</font> Guangrun Wang, Yansong Tang, Liang Lin, Philip H.S. Torr \
    > <font color=Gray>**Organization(s)**:</font> University of Oxford; Tsinghua-Berkeley Shenzhen Institute, Tsinghua University; Sun Yat-sen University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Rethinking the Augmentation Module in Contrastive Learning: Learning Hierarchical Augmentation Invariance with Expanded Views (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.pdf) [[~~No code~~]]()
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

- Exploring the Equivalence of Siamese Self-Supervised Learning via A Unified Gradient Framework (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Tao_Exploring_the_Equivalence_of_Siamese_Self-Supervised_Learning_via_a_Unified_CVPR_2022_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Tao_Exploring_the_Equivalence_CVPR_2022_supplemental.pdf)
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

- Exploring Set Similarity for Dense Self-supervised Representation Learning (**SetSim** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Align_Representations_With_Base_A_New_Approach_to_Self-Supervised_Learning_CVPR_2022_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Shaofeng Zhang, Lyn Qiu, Feng Zhu, Junchi Yan, Hengrui Zhang, Rui Zhao, Hongyang Li, Xiaokang Yang \
    > <font color=Gray>**Organization(s)**:</font>  University of Sydney; Kuaishou Technology; Xidian University; University of Melbourne\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Local representation

- Patch-level Representation Learning for Self-supervised Vision Transformers (**SelfPatch** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yun_Patch-Level_Representation_Learning_for_Self-Supervised_Vision_Transformers_CVPR_2022_paper.pdf) [[code]](https://github.com/alinlab/SelfPatch) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Yun_Patch-Level_Representation_Learning_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Sukmin Yun, Hankook Lee, Jaehyung Kim, Jinwoo Shin \
    > <font color=Gray>**Organization(s)**:</font>  Korea Advanced Institute of Science and Technology (KAIST)\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Local representation

- Self-Supervised Learning of Object Parts for Semantic Segmentation (**Leopart** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf) [[~~No code~~]]( https://github.com/MkuuWaUjinga/leopart) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Ziegler_Self-Supervised_Learning_of_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Adrian Ziegler, Yuki M. Asano \
    > <font color=Gray>**Organization(s)**:</font>  Technical University of Munich; QUVA Lab, University of Amsterdam\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Local representation

- Point-Level Region Contrast for Object Detection Pre-Training (**** - <font color="#dd0000">**CVPR**22</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Bai_Point-Level_Region_Contrast_for_Object_Detection_Pre-Training_CVPR_2022_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Bai_Point-Level_Region_Contrast_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yutong Bai, Xinlei Chen, Alexander Kirillov, Alan Yuille, Alexander C. Berg \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research (FAIR); Johns Hopkins University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Local representation

- Use All The Labels: A Hierarchical Multi-Label Contrastive Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Use_All_the_Labels_A_Hierarchical_Multi-Label_Contrastive_Learning_Framework_CVPR_2022_paper.pdf) [[code]](https://github.com/salesforce/hierarchicalContrastiveLearning) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_Use_All_the_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Shu Zhang, Ran Xu, Caiming Xiong, Chetan Ramaiah \
    > <font color=Gray>**Organization(s)**:</font>  Salesforce Research\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Label supervision
    
- Revisiting the Transferability of Supervised Pretraining: an MLP Perspective (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Revisiting_the_Transferability_of_Supervised_Pretraining_An_MLP_Perspective_CVPR_2022_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Revisiting_the_Transferability_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yizhou Wang, Shixiang Tang, Feng Zhu, Lei Bai, Rui Zhao, Donglian Qi, Wanli Ouyang \
    > <font color=Gray>**Organization(s)**:</font>  Zhejiang University; The University of Sydney; SenseTime Research; Shanghai Jiao Tong University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Label supervision

- Does Robustness on ImageNet Transfer to Downstream Tasks? (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yamada_Does_Robustness_on_ImageNet_Transfer_to_Downstream_Tasks_CVPR_2022_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Yamada_Does_Robustness_on_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yutaro Yamada, Mayu Otani \
    > <font color=Gray>**Organization(s)**:</font>  Yale University; CyberAgent, Inc. \
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- When Does Contrastive Visual Representation Learning Work? (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Cole_When_Does_Contrastive_Visual_Representation_Learning_Work_CVPR_2022_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Elijah Cole, Xuan Yang, Kimberly Wilber, Oisin Mac Aodha, Serge Belongie \
    > <font color=Gray>**Organization(s)**:</font>  Caltech; Google; University of Edinburgh; Alan Turing Institute; University of Copenhagen\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- Rethinking Minimal Sufficient Representation in Contrastive Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Rethinking_Minimal_Sufficient_Representation_in_Contrastive_Learning_CVPR_2022_paper.pdf) [[code]](https://github.com/Haoqing-Wang/InfoCL) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Rethinking_Minimal_Sufficient_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Haoqing Wang, Xun Guo, Zhi-Hong Deng, Yan Lu \
    > <font color=Gray>**Organization(s)**:</font>  Peking University; Microsoft Research Asia\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- Beyond Supervised vs. Unsupervised: Representative Benchmarking and Analysis of Image Representation Learning (**** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gwilliam_Beyond_Supervised_vs._Unsupervised_Representative_Benchmarking_and_Analysis_of_Image_CVPR_2022_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Gwilliam_Beyond_Supervised_vs._CVPR_2022_supplemental.pdf)
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

- Masked Feature Prediction for Self-Supervised Visual Pre-Training (**MaskFeat** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wei_Masked_Feature_Prediction_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, Christoph Feichtenhofer \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research; Johns Hopkins University\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Transformer

- Self-Taught Metric Learning without Labels (**STML** - <font color="#dd0000">**CVPR**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Self-Taught_Metric_Learning_Without_Labels_CVPR_2022_paper.pdf) [[code]](https://github.com/tjddus9597/STML-CVPR22) [[supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Kim_Self-Taught_Metric_Learning_CVPR_2022_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Sungyeon Kim, Dongwon Kim, Minsu Cho, Suha Kwak \
    > <font color=Gray>**Organization(s)**:</font>  FDept. of CSE, POSTECH; Graduate School of AI, POSTECH\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- CoDo: Contrastive Learning with Downstream Background Invariance for Detection (**CoDo** - <font color="#dd0000">**CVPRW**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Zhao_CoDo_Contrastive_Learning_With_Downstream_Background_Invariance_for_Detection_CVPRW_2022_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Bing Zhao, Jun Li, Hong Zhu \
    > <font color=Gray>**Organization(s)**:</font>  Department of AI and HPC Inspur Electronic Information Industry Co., Ltd\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Local representation

- What Should Be Equivariant In Self-Supervised Learning (**** - <font color="#dd0000">**CVPRW**22</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Xie_What_Should_Be_Equivariant_in_Self-Supervised_Learning_CVPRW_2022_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Yuyang Xie, Jianhong Wen, Kin Wai Lau, Yasar Abbas Ur Rehman, Jiajun Shen \
    > <font color=Gray>**Organization(s)**:</font>  TCL AI Lab; Fuzhou University; City University of Hong Kong\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> Analysis

- Where are my Neighbors? Exploiting Patches Relations in Self-Supervised Vision Transformer (**** - <font color="#dd0000">**CVPRW**22</font>?) [[paper]](https://arxiv.org/abs/2206.00481) [[code]](https://github.com/guglielmocamporese/relvit)
    > <font color=Gray>**Author(s)**:</font> Guglielmo Camporese, Elena Izzo, Lamberto Ballan \
    > <font color=Gray>**Organization(s)**:</font>  University of Padova, Italy\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Deeply Unsupervised Patch Re-Identification for Pre-training Object Detectors (**DUPR** - <font color="#dd0000">**TPAMI**22</font>) [[paper]](https://ieeexplore.ieee.org/document/9749837) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Jian Ding, Enze Xie, Hang Xu, Chenhan Jiang, Zhenguo Li, Ping Luo, Guidong Xia \
    > <font color=Gray>**Organization(s)**:</font> Wuhan University; Huawei Noah’s Ark Lab; University of Hong Kong\
    > <font color=Gray>**Description**:</font>  \
    > <font color=Gray>**Tags**:</font> 

- Learning Generalized Transformation Equivariant Representations Via AutoEncoding Transformations (**** - <font color="#dd0000">**TPAMI**22</font>) [[paper]](https://ieeexplore.ieee.org/document/9219238) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Guo-Jun Qi, Liheng Zhang, Feng Lin, Xiao Wang \
    > <font color=Gray>**Organization(s)**:</font> Futurewei Seattle Cloud Lab; University of Central Florida; University of Science and Technology of China; Purdue University \
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

- Spatial Assembly Networks for Image Representation Learning (**SAN** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spatial_Assembly_Networks_for_Image_Representation_Learning_CVPR_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Li_Spatial_Assembly_Networks_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yang Li, Shichao Kan, Jianhe Yuan, Wenming Cao, Zhihai He \
    > <font color=Gray>**Organization(s)**:</font>  University of Missouri, MO; Beijing Jiaotong University; Shenzhen University\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- SelfAugment: Automatic Augmentation Policies for Self-Supervised Learning (**SelfAugment** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Reed_SelfAugment_Automatic_Augmentation_Policies_for_Self-Supervised_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/cjrd/selfaugment) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Reed_SelfAugment_Automatic_Augmentation_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Colorado J Reed, Sean Metzger, Aravind Srinivas, Trevor Darrell, Kurt Keutzer \
    > <font color=Gray>**Organization(s)**:</font>  BAIR, Department of Computer Science, UC Berkeley; Graduate Group in Bioengineering (Berkeley/UCSF), Weill Neurosciences Institute & UCSF Neurological Surgery \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Progressive Stage-wise Learning for Unsupervised Feature Representation Enhancement (**PSL** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Stage-Wise_Learning_for_Unsupervised_Feature_Representation_Enhancement_CVPR_2021_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Zefan Li, Chenxi Liu, Alan Yuille, Bingbing Ni, Wenjun Zhang, Wen Gao \
    > <font color=Gray>**Organization(s)**:</font>  Shanghai Jiao Tong University; Johns Hopkins University; Peking University; MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning (**PixPro** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Propagate_Yourself_Exploring_Pixel-Level_Consistency_for_Unsupervised_Visual_Representation_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/zdaxie/PixPro)
    > <font color=Gray>**Author(s)**:</font> Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, Han Hu \
    > <font color=Gray>**Organization(s)**:</font>  Tsinghua University; Xi’an Jiaotong University; Microsoft Research Asia \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Local representation

- How Well Do Self-Supervised Models Transfer? (**** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ericsson_How_Well_Do_Self-Supervised_Models_Transfer_CVPR_2021_paper.pdf) [[code]](https://github.com/linusericsson/ssl-transfer) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Ericsson_How_Well_Do_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Linus Ericsson, Henry Gouk, Timothy M. Hospedales \
    > <font color=Gray>**Organization(s)**:</font>  University of Edinburgh; Samsung AI Research, Cambridge \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Analysis

- Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning (**EMAN** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Exponential_Moving_Average_Normalization_for_Self-Supervised_and_Semi-Supervised_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/amazon-research/exponential-moving-average-normalization) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Cai_Exponential_Moving_Average_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Zhaowei Cai, Avinash Ravichandran, Subhransu Maji, Charless Fowlkes, Zhuowen Tu, Stefano Soatto \
    > <font color=Gray>**Organization(s)**:</font>  Amazon Web Services \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Dense Contrastive Learning for Self-Supervised Visual Pre-Training (**DenseCL** - <font color="#dd0000">**CVPR**21</font> Oral) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Dense_Contrastive_Learning_for_Self-Supervised_Visual_Pre-Training_CVPR_2021_paper.pdf) [[code]](https://github.com/WXinlong/DenseCL) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Dense_Contrastive_Learning_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, Lei Li \
    > <font color=Gray>**Organization(s)**:</font>  The University of Adelaide; Tongji University; ByteDance AI Lab \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Local representation

- Unsupervised Feature Learning by Cross-Level Instance-Group Discrimination (**CLD** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Feature_Learning_by_Cross-Level_Instance-Group_Discrimination_CVPR_2021_paper.pdf) [[code]]( https://github.com/frankxwang/CLD-UnsupervisedLearning) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Unsupervised_Feature_Learning_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Xudong Wang, Ziwei Liu, Stella X. Yu \
    > <font color=Gray>**Organization(s)**:</font>  UC Berkeley / ICSI; S-Lab, NTU \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries (**AdCo** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_AdCo_Adversarial_Contrast_for_Efficient_Learning_of_Unsupervised_Representations_From_CVPR_2021_paper.pdf) [[code]](https://github.com/maple-research-lab/AdCo) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Hu_AdCo_Adversarial_Contrast_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Qianjiang Hu, Xiao Wang, Wei Hu, Guo-Jun Qi \
    > <font color=Gray>**Organization(s)**:</font>  Peking University; Purdue University; Laboratory for MAchine Perception and LEarning (MAPLE) \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- CASTing Your Model: Learning to Localize Improves Self-Supervised Representations (**CAST** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Selvaraju_CASTing_Your_Model_Learning_To_Localize_Improves_Self-Supervised_Representations_CVPR_2021_paper.pdf) [[code]](https://github.com/salesforce/CAST) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Selvaraju_CASTing_Your_Model_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Ramprasaath R. Selvaraju, Karan Desai, Justin Johnson, Nikhil Naik \
    > <font color=Gray>**Organization(s)**:</font> Salesforce Research; University of Michigan \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Instance Localization for Self-supervised Detection Pretraining (**InsLoc** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Instance_Localization_for_Self-Supervised_Detection_Pretraining_CVPR_2021_paper.pdf) [[code]](https://github.com/limbo0000/InstanceLoc) 
    > <font color=Gray>**Author(s)**:</font> Ceyuan Yang, Zhirong Wu, Bolei Zhou, Stephen Lin \
    > <font color=Gray>**Organization(s)**:</font> Chinese University of Hong Kong; Microsoft Research Asia \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- PiCIE: Unsupervised Semantic Segmentation using Invariance and Equivariance in Clustering (**PiCIE** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Cho_PiCIE_Unsupervised_Semantic_Segmentation_Using_Invariance_and_Equivariance_in_Clustering_CVPR_2021_paper.pdf) [[code]](https://github.com/janghyuncho/PiCIE) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Cho_PiCIE_Unsupervised_Semantic_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Jang Hyun Cho, Utkarsh Mall, Kavita Bala, Bharath Hariharan \
    > <font color=Gray>**Organization(s)**:</font> University of Texas at Austin; Cornell University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  Unsupervised semantic segmentation

- Self-supervised Augmentation Consistency for Adapting Semantic Segmentation (**SAC** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Araslanov_Self-Supervised_Augmentation_Consistency_for_Adapting_Semantic_Segmentation_CVPR_2021_paper.pdf) [[code]](https://github.com/visinf/da-sac) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Araslanov_Self-Supervised_Augmentation_Consistency_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Nikita Araslanov, Stefan Roth \
    > <font color=Gray>**Organization(s)**:</font> Department of Computer Science, TU Darmstadt; hessian.AI \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  Domain adaptation

- Spatially Consistent Representation Learning (**SCRL** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Roh_Spatially_Consistent_Representation_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/kakaobrain/scrl) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Roh_Spatially_Consistent_Representation_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Byungseok Roh, Wuhyun Shin, Ildoo Kim, Sungwoong Kim \
    > <font color=Gray>**Organization(s)**:</font> Kakao Brain \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  Local representation

- UP-DETR: Unsupervised Pre-training for Object Detection with Transformers (**UP-DETR** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_UP-DETR_Unsupervised_Pre-Training_for_Object_Detection_With_Transformers_CVPR_2021_paper.pdf) [[code]](https://github.com/dddzg/up-detr) [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Dai_UP-DETR_Unsupervised_Pre-Training_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Zhigang Dai, Bolun Cai, Yugeng Lin, Junying Chen \
    > <font color=Gray>**Organization(s)**:</font> School of Software Engineering, South China University of Technology; Tencent Wechat AI; Key Laboratory of Big Data and Intelligent Robot \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Understanding the Behaviour of Contrastive Loss (**** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Wang_Understanding_the_Behaviour_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Feng Wang, Huaping Liu\
    > <font color=Gray>**Organization(s)**:</font> Beijing National Research Center for Information Science and Technology(BNRist), Department of Computer Science and Technology, Tsinghua University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Analysis

- Unsupervised Hyperbolic Metric Learning (**** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Unsupervised_Hyperbolic_Metric_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/JiexiYan/UnHyperML) 
    > <font color=Gray>**Author(s)**:</font> Jiexi Yan, Lei Luo, Cheng Deng, Heng Huang \
    > <font color=Gray>**Organization(s)**:</font> Xidian University; University of Pittsburgh; JD Finance America Corporation, Mountain View \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Metric Learning

- Relative Order Analysis and Optimization for Unsupervised Deep Metric Learning (**ROUL** - <font color="#dd0000">**CVPR**21</font>) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Kan_Relative_Order_Analysis_and_Optimization_for_Unsupervised_Deep_Metric_Learning_CVPR_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Kan_Relative_Order_Analysis_CVPR_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Shichao Kan, Yigang Cen, Yang Li, Vladimir Mladenovic, Zhihai He \
    > <font color=Gray>**Organization(s)**:</font>  Beijing Jiaotong University; Beijing Key Laboratory of Advanced Information Science and Network Technology; University of Missouri; Faculty of Technical Sciences University of Kragujevac \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Metric Learning

- DetCo: Unsupervised Contrastive Learning for Object Detection (**DetCo** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_DetCo_Unsupervised_Contrastive_Learning_for_Object_Detection_ICCV_2021_paper.pdf) [[code]](github.com/xieenze/DetCo) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Xie_DetCo_Unsupervised_Contrastive_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Enze Xie, Jian Ding, Wenhai Wang, Xiaohang Zhan, Hang Xu, Peize Sun, Zhenguo Li, Ping Luo \
    > <font color=Gray>**Organization(s)**:</font> The University of Hong Kong; Huawei Noah’s Ark Lab; Wuhan University; Nanjing University; Chinese University of Hong Kong  \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Divide and Contrast: Self-supervised Learning from Uncurated Data (**DnC** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Divide_and_Contrast_Self-Supervised_Learning_From_Uncurated_Data_ICCV_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Tian_Divide_and_Contrast_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yonglong Tian, Olivier J. Hénaff, Aäron van den Oord \
    > <font color=Gray>**Organization(s)**:</font> MIT; DeepMind \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Uncurated data

- Improve Unsupervised Pretraining for Few-label Transfer (**** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Improve_Unsupervised_Pretraining_for_Few-Label_Transfer_ICCV_2021_paper.pdf) [[~~No code~~]]() 
    > <font color=Gray>**Author(s)**:</font> Suichan Li, Dongdong Chen, Yinpeng Chen, Lu Yuan, Lei Zhang, Qi Chu, Bin Liu, Nenghai Yu \
    > <font color=Gray>**Organization(s)**:</font> University of Science and Technology of China; Microsoft Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervised Visual Representations Learning by Contrastive Mask Prediction (**MaskCo** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Self-Supervised_Visual_Representations_Learning_by_Contrastive_Mask_Prediction_ICCV_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Zhao_Self-Supervised_Visual_Representations_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Yucheng Zhao, Guangting Wang, Chong Luo, Wenjun Zeng, Zheng-Jun Zha \
    > <font color=Gray>**Organization(s)**:</font> University of Science and Technology of China; Microsoft Research Asia \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Local representation

- Mean Shift for Self-Supervised Learning (**MSF** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Koohpayegani_Mean_Shift_for_Self-Supervised_Learning_ICCV_2021_paper.pdf) [[code]](https://github.com/UMBCvision/MSF) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Koohpayegani_Mean_Shift_for_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Soroush Abbasi Koohpayegani, Ajinkya Tejankar, Hamed Pirsiavash \
    > <font color=Gray>**Organization(s)**:</font> University of Maryland, Baltimore County; University of California, Davis \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Positive discovery

- On Compositions of Transformations in Contrastive Self-Supervised Learning (**GDT** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Patrick_On_Compositions_of_Transformations_in_Contrastive_Self-Supervised_Learning_ICCV_2021_paper.pdf) [[code]](https://github.com/facebookresearch/GDT) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Patrick_On_Compositions_of_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Mandela Patrick, Yuki M. Asano, Polina Kuznetsova, Ruth Fong, João F. Henriques, Geoffrey Zweig, Andrea Vedaldi \
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research; Visual Geometry Group, University of Oxford \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- On Feature Decorrelation in Self-Supervised Learning (**Shuffled-DBN** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hua_On_Feature_Decorrelation_in_Self-Supervised_Learning_ICCV_2021_paper.pdf) [[~~No code~~]]() 
    > <font color=Gray>**Author(s)**:</font> Tianyu Hua, Wenxiao Wang, Zihui Xue, Sucheng Ren, Yue Wang, Hang Zhao \
    > <font color=Gray>**Organization(s)**:</font> Tsinghua University; Shanghai Qi Zhi Institute; UT Austin; MIT; South China University of Technology \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-supervised Product Quantization for Deep Unsupervised Image Retrieval (**SPQ** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_Self-Supervised_Product_Quantization_for_Deep_Unsupervised_Image_Retrieval_ICCV_2021_paper.pdf) [[code]](https://github.com/youngkyunJang/SPQ) 
    > <font color=Gray>**Author(s)**:</font> Young Kyun Jang, Nam Ik Cho \
    > <font color=Gray>**Organization(s)**:</font> Department of ECE, INMC, Seoul National University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations (**NNCLR** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Dwibedi_With_a_Little_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, Andrew Zisserman \
    > <font color=Gray>**Organization(s)**:</font> Google Research; DeepMind \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Contrasting Contrastive Self-Supervised Representation Learning Pipelines (**ViRB** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kotar_Contrasting_Contrastive_Self-Supervised_Representation_Learning_Pipelines_ICCV_2021_paper.pdf) [[code]](https://github.com/allenai/virb) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Kotar_Contrasting_Contrastive_Self-Supervised_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Klemen Kotar, Gabriel Ilharco, Ludwig Schmidt, Kiana Ehsani, Roozbeh Mottaghi \
    > <font color=Gray>**Organization(s)**:</font> PRIOR @ Allen Institute for AI; University of Washington \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Analysis

- ISD: Self-Supervised Learning by Iterative Similarity Distillation (**ISD** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Tejankar_ISD_Self-Supervised_Learning_by_Iterative_Similarity_Distillation_ICCV_2021_paper.pdf) [[code]](https://github.com/UMBCvision/ISD) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Tejankar_ISD_Self-Supervised_Learning_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Ajinkya Tejankar, Soroush Abbasi Koohpayegani, Vipin Pillai, Paolo Favaro, Hamed Pirsiavash \
    > <font color=Gray>**Organization(s)**:</font> University of Maryland, Baltimore County; University of Bern; University of California, Davis \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- CDS: Cross-Domain Self-supervised Pre-training (**CDS** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_CDS_Cross-Domain_Self-Supervised_Pre-Training_ICCV_2021_paper.pdf) [[code]](https://github.com/VisionLearningGroup/CDS) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Kim_CDS_Cross-Domain_Self-Supervised_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Donghyun Kim, Kuniaki Saito, Tae-Hyun Oh, Bryan A. Plummer, Stan Sclaroff, Kate Saenko \
    > <font color=Gray>**Organization(s)**:</font> Boston University; POSTECH; MIT-IBM Watson AI Lab \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Collaborative Unsupervised Visual Representation Learning from Decentralized Data (**FedU** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhuang_Collaborative_Unsupervised_Visual_Representation_Learning_From_Decentralized_Data_ICCV_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Zhuang_Collaborative_Unsupervised_Visual_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Weiming Zhuang, Xin Gan, Yonggang Wen, Shuai Zhang, Shuai Yi \
    > <font color=Gray>**Organization(s)**:</font> S-Lab, Nanyang Technological University; Nanyang Technological University; SenseTime Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Weakly Supervised Contrastive Learning (**WCL** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Weakly_Supervised_Contrastive_Learning_ICCV_2021_paper.pdf) [[code]](https://github.com/KyleZheng1997/WCL) 
    > <font color=Gray>**Author(s)**:</font> Mingkai Zheng, Fei Wang, Shan You, Chen Qian, Changshui Zhang, Xiaogang Wang, Chang Xu \
    > <font color=Gray>**Organization(s)**:</font> SenseTime Research; University of Science and Technology of China; Tsinghua University; The Chinese University of Hong Kong; The University of Sydney \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervised Representation Learning from Flow Equivariance (**FlowE** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiong_Self-Supervised_Representation_Learning_From_Flow_Equivariance_ICCV_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Xiong_Self-Supervised_Representation_Learning_ICCV_2021_supplemental.zip)
    > <font color=Gray>**Author(s)**:</font> Yuwen Xiong, Mengye Ren, Wenyuan Zeng, Raquel Urtasun \
    > <font color=Gray>**Organization(s)**:</font> Waabi; University of Toronto \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Vi2CLR: Video and Image for Visual Contrastive Learning of Representation (**Vi2CLR** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Diba_Vi2CLR_Video_and_Image_for_Visual_Contrastive_Learning_of_Representation_ICCV_2021_paper.pdf) [[~~No code~~]]() 
    > <font color=Gray>**Author(s)**:</font> Ali Diba, Vivek Sharma, Reza Safdari, Dariush Lotfi, Saquib Sarfraz, Rainer Stiefelhagen, Luc Van Gool \
    > <font color=Gray>**Organization(s)**:</font>  KU Leuven; Sensifai; Karlsruhe Institute of Technology; Daimler TSS; Massachusetts Institute of Technology; Harvard Medical School; ETH Zurich \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Solving Inefficiency of Self-supervised Representation Learning (**truncated triplet** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Solving_Inefficiency_of_Self-Supervised_Representation_Learning_ICCV_2021_paper.pdf) [[code]](https://github.com/wanggrun/triplet) 
    > <font color=Gray>**Author(s)**:</font> Guangrun Wang, Keze Wang, Guangcong Wang, Philip H.S. Torr, Liang Lin \
    > <font color=Gray>**Organization(s)**:</font>  Sun Yat-sen University; University of Oxford; Nanyang Technological University; DarkMatter AI Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Instance Similarity Learning for Unsupervised Feature Representation (**ISL** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Instance_Similarity_Learning_for_Unsupervised_Feature_Representation_ICCV_2021_paper.pdf) [[code]](https://github.com/ZiweiWangTHU/ISL.git) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wang_Instance_Similarity_Learning_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Ziwei Wang, Yunsong Wang, Ziyi Wu, Jiwen Lu, Jie Zhou \
    > <font color=Gray>**Organization(s)**:</font>  Tsinghua University; State Key Lab of Intelligent Technologies and Systems; Beijing National Research Center for Information Science and Technology \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Improving Contrastive Learning by Visualizing Feature Transformation (**** - <font color="#dd0000">**ICCV**21</font> Oral) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Improving_Contrastive_Learning_by_Visualizing_Feature_Transformation_ICCV_2021_paper.pdf) [[code]](https://github.com/DTennant/CL-Visualizing-Feature-Transformation) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Zhu_Improving_Contrastive_Learning_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Rui Zhu, Bingchen Zhao, Jingen Liu, Zhenglong Sun, Chang Wen Chen \
    > <font color=Gray>**Organization(s)**:</font>   The Chinese University of HongKong, Shenzhen; JD AI Research; Tongji University; The Hong Kong Polytechnic University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Temporal Knowledge Consistency for Unsupervised Visual Representation Learning (**TKC** - <font color="#dd0000">**ICCV**21</font> Oral) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_Temporal_Knowledge_Consistency_for_Unsupervised_Visual_Representation_Learning_ICCV_2021_paper.pdf) [[~~No code~~]]() [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Feng_Temporal_Knowledge_Consistency_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Weixin Feng, Yuanjiang Wang, Lihua Ma, Ye Yuan, Chi Zhang \
    > <font color=Gray>**Organization(s)**:</font>   Beijing University of Posts and Telecommunications; Megvii Technology \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- MultiSiam: Self-supervised Multi-instance Siamese Representation Learning for Autonomous Driving (**MultiSiam** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_MultiSiam_Self-Supervised_Multi-Instance_Siamese_Representation_Learning_for_Autonomous_Driving_ICCV_2021_paper.pdf) [[code]](https://github.com/KaiChen1998/MultiSiam) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Chen_MultiSiam_Self-Supervised_Multi-Instance_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Kai Chen, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung \
    > <font color=Gray>**Organization(s)**:</font> Hong Kong University of Science and Technology; Huawei Noah’s Ark Lab   \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Local representation

- Region Similarity Representation Learning(**ReSim** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiao_Region_Similarity_Representation_Learning_ICCV_2021_paper.pdf) [[code]](https://github.com/Tete-Xiao/ReSim) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Xiao_Region_Similarity_Representation_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Tete Xiao, Colorado J Reed, Xiaolong Wang, Kurt Keutzer, Trevor Darrell \
    > <font color=Gray>**Organization(s)**:</font> UC Berkeley; UC San Diego   \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- An Empirical Study of Training Self-Supervised Vision Transformers(**MoCo-v3** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [[unofficial code]](https://github.com/open-mmlab/mmselfsup) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Chen_An_Empirical_Study_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Xinlei Chen, Saining Xie, Kaiming He \
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research (FAIR)  \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Emerging Properties in Self-Supervised Vision Transformers(**DINO** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf) [[code]]( https://github.com/facebookresearch/dino) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Caron_Emerging_Properties_in_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin \
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research; Inria∗; Sorbonne University  \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- A Broad Study on the Transferability of Visual Representations with Contrastive Learning (**** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Islam_A_Broad_Study_on_the_Transferability_of_Visual_Representations_With_ICCV_2021_paper.pdf) [[code]](https://github.com/asrafulashiq/transfer_broad) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Islam_A_Broad_Study_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Ashraful Islam, Chun-Fu (Richard) Chen, Rameswar Panda, Leonid Karlinsky, Richard Radke, Rogerio Feris \
    > <font color=Gray>**Organization(s)**:</font>   Rensselaer Polytechnic Institute; MIT-IBM Watson AI Lab; IBM Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

- Concept Generalization in Visual Representation Learning (**CoG** - <font color="#dd0000">**ICCV**21</font>) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Sariyildiz_Concept_Generalization_in_Visual_Representation_Learning_ICCV_2021_paper.pdf) [[code]](https://europe.naverlabs.com/cog-benchmark) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Sariyildiz_Concept_Generalization_in_ICCV_2021_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Mert Bulent Sariyildiz, Yannis Kalantidis, Diane Larlus, Karteek Alahari \
    > <font color=Gray>**Organization(s)**:</font>   NAVER LABS Europe; Inria \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>
    
- Concurrent Discrimination and Alignment for Self-Supervised Feature Learning (**CODIAL** - <font color="#dd0000">**ICCVW**21</font>) [[paper]](https://ieeexplore.ieee.org/document/9607472) [[code]](https://github.com/AnjanDutta/codial)
    > <font color=Gray>**Author(s)**:</font> Anjan Dutta, Massimiliano Mancini, Zeynep Akata \
    > <font color=Gray>**Organization(s)**:</font>   University of Exeter; University of T¨ubingen \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

## 2020

- Momentum Contrast for Unsupervised Visual Representation Learning (**MoCo** - <font color="#dd0000">**CVPR**20</font> Oral) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) [[code]](https://github.com/facebookresearch/moco) [[supp]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/He_Momentum_Contrast_for_CVPR_2020_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research (FAIR)\
    > <font color=Gray>**Description**:</font>  Building a dynamic dictionary with a queue and a moving-averaged encoder \
    > <font color=Gray>**Tags**:</font> InfoNCE loss, siamese architecture

- Online Deep Clustering for Unsupervised Representation Learning (**ODC** - <font color="#dd0000">**CVPR**20</font> Oral) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf) [[code]](https://github.com/open-mmlab/mmselfsup) 
    > <font color=Gray>**Author(s)**:</font> Xiaohang Zhan, Jiahao Xie, Ziwei Liu, Yew-Soon Ong, Chen Change Loy \
    > <font color=Gray>**Organization(s)**:</font> CUHK - SenseTime Joint Lab, The Chinese University of Hong Kong; Nanyang Technological University; AI3, A*STAR \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics (**LCI** - <font color="#dd0000">**CVPR**20</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jenni_Steering_Self-Supervised_Feature_Learning_Beyond_Local_Pixel_Statistics_CVPR_2020_paper.pdf) [[code]](https://github.com/sjenni/LCI) [[supp]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Jenni_Steering_Self-Supervised_Feature_CVPR_2020_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Simon Jenni, Hailin Jin, Paolo Favaro \
    > <font color=Gray>**Organization(s)**:</font> University of Bern; Adobe Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Learning Representations by Predicting Bags of Visual Words (**BoWNet** - <font color="#dd0000">**CVPR**20</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gidaris_Learning_Representations_by_Predicting_Bags_of_Visual_Words_CVPR_2020_paper.pdf) [[code]](https://github.com/valeoai/bownet) [[supp]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Gidaris_Learning_Representations_by_CVPR_2020_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Spyros Gidaris, Andrei Bursuc, Nikos Komodakis, Patrick Perez, Matthieu Cord \
    > <font color=Gray>**Organization(s)**:</font> Valeo.ai; University of Crete \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Probabilistic Structural Latent Representation for Unsupervised Embedding (**PSLR** - <font color="#dd0000">**CVPR**20</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Probabilistic_Structural_Latent_Representation_for_Unsupervised_Embedding_CVPR_2020_paper.pdf) [[code]](https://github.com/mangye16/PSLR) 
    > <font color=Gray>**Author(s)**:</font> Mang Ye, Jianbing Shen \
    > <font color=Gray>**Organization(s)**:</font> Inception Institute of Artificial Intelligence \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervised Learning of Pretext-Invariant Representations (**PIRL** - <font color="#dd0000">**CVPR**20</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.pdf) [[unofficial code]](https://github.com/akwasigroch/Pretext-Invariant-Representations) 
    > <font color=Gray>**Author(s)**:</font> Ishan Misra, Laurens van der Maaten \
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- How Useful is Self-Supervised Pretraining for Visual Tasks? (**** - <font color="#dd0000">**CVPR**20</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Newell_How_Useful_Is_Self-Supervised_Pretraining_for_Visual_Tasks_CVPR_2020_paper.pdf) [[code]](github.com/princeton-vl/selfstudy) [[supp]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Newell_How_Useful_Is_CVPR_2020_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Alejandro Newell, Jia Deng \
    > <font color=Gray>**Organization(s)**:</font> Princeton University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  Analysis

- SCAN: Learning to Classify Images without Labels (**SCAN** - <font color="#dd0000">**ECCV**20</font>) [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_16) [[code]](https://github.com/wvangansbeke/Unsupervised-Classification) [[arxiv]](https://arxiv.org/abs/2005.12320)
    > <font color=Gray>**Author(s)**:</font> Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, Luc Van Gool \
    > <font color=Gray>**Organization(s)**:</font> KU Leuven/ESAT-PSI; ETH Zurich/CVL, TRACE \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  

- Contrastive Multiview Coding (**CMC** - <font color="#dd0000">**ECCV**20</font>) [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_45) [[code]](http://github.com/HobbitLong/CMC/) [[arxiv]](https://arxiv.org/abs/1906.05849)
    > <font color=Gray>**Author(s)**:</font> Yonglong Tian, Dilip Krishnan, Phillip Isola \
    > <font color=Gray>**Organization(s)**:</font> MIT CSAIL; Google Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  

- Mitigating Embedding and Class Assignment Mismatch in Unsupervised Image Classification (**** - <font color="#dd0000">**ECCV**20</font>) [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690749.pdf) [[code]](https://github.com/dscig/TwoStageUC) [[~~No arxiv~~]]()
    > <font color=Gray>**Author(s)**:</font> 	Sungwon Han, Sungwon Park, Sungkyu Park, Sundong Kim, Meeyoung Cha \
    > <font color=Gray>**Organization(s)**:</font> Korea Advanced Institute of Science and Technology; Data Science Group, Institute for Basic Science \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  

- Unsupervised Deep Metric Learning with Transformed Attention Consistency and Contrastive Clustering Loss (**** - <font color="#dd0000">**ECCV**20</font>) [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560137.pdf) [[~~No code~~]]() 
    > <font color=Gray>**Author(s)**:</font> Yang Li, Shichao Kan, Zhihai He	 \
    > <font color=Gray>**Organization(s)**:</font> University of Missouri; Beijing Jiaotong University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>  

- Unsupervised Image Classification for Deep
Representation Learning (**** - <font color="#dd0000">**ECCVW**20</font>) [[paper]](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_30) [[~~No code~~]]() [[arxiv]](https://arxiv.org/abs/2006.11480)
    > <font color=Gray>**Author(s)**:</font> 	Weijie Chen, Shiliang Pu, Di Xie, Shicai Yang, Yilu Guo, Luojun Lin \
    > <font color=Gray>**Organization(s)**:</font> Hikvision Research Institute; School of Electronic and Information Engineering, South China University of Technology \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 


## 2019

- Unsupervised Embedding Learning via Invariant and Spreading Instance Feature (**ISIF** - <font color="#dd0000">**CVPR**19</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ye_Unsupervised_Embedding_Learning_via_Invariant_and_Spreading_Instance_Feature_CVPR_2019_paper.pdf) [[code]](https://github.com/mangye16/Unsupervised_Embedding_Learning)
    > <font color=Gray>**Author(s)**:</font> Mang Ye, Xu Zhang,  Pong C. Yuen, Shih-Fu Chang\
    > <font color=Gray>**Organization(s)**:</font>  Hong Kong Baptist University; Columbia University\
    > <font color=Gray>**Description**:</font>  Learning augmentation invariant  and instance spread-out features \
    > <font color=Gray>**Tags**:</font> Siamese architecture

- Self-Supervised Representation Learning by Rotation Feature Decoupling (**FeatureDecoupling** - <font color="#dd0000">**CVPR**19</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf) [[code]](https://github.com/philiptheother/FeatureDecoupling) [[supp]](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Feng_Self-Supervised_Representation_Learning_CVPR_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Zeyu Feng, Chang Xu, Dacheng Tao \
    > <font color=Gray>**Organization(s)**:</font> UBTECH Sydney AI Centre, School of Computer Science; University of Sydney \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Iterative Reorganization with Weak Spatial Constraints: Solving Arbitrary Jigsaw Puzzles for Unsupervised Representation Learning (**** - <font color="#dd0000">**CVPR**19</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Iterative_Reorganization_With_Weak_Spatial_Constraints_Solving_Arbitrary_Jigsaw_Puzzles_CVPR_2019_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Chen Wei, Lingxi Xie, Xutong Ren, Yingda Xia, Chi Su, Jiaying Liu, Qi Tian, Alan L. Yuille \
    > <font color=Gray>**Organization(s)**:</font> Peking University; The Johns Hopkins University; Kingsoft; Huawei Noah’s Ark Lab \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervised GANs via Auxiliary Rotation Loss (**SS-GAN** - <font color="#dd0000">**CVPR**19</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Self-Supervised_GANs_via_Auxiliary_Rotation_Loss_CVPR_2019_paper.pdf) [[code]]( https://github.com/google/compare_gan) [[supp]](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Chen_Self-Supervised_GANs_via_CVPR_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Ting Chen, Xiaohua Zhai, Marvin Ritter, Mario Lucic, Neil Houlsby \
    > <font color=Gray>**Organization(s)**:</font> University of California, Los Angeles; Google Brain \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> GAN

- AET vs. AED: Unsupervised Representation Learning by Auto-Encoding Transformations rather than Data (**AET** - <font color="#dd0000">**CVPR**19</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_AET_vs._AED_Unsupervised_Representation_Learning_by_Auto-Encoding_Transformations_Rather_CVPR_2019_paper.pdf) [[code]](https://github.com/maple-research-lab/AET)
    > <font color=Gray>**Author(s)**:</font> Liheng Zhang, Guo-Jun Qi, Liqiang Wang, Jiebo Luo \
    > <font color=Gray>**Organization(s)**:</font> Huawei Cloud; University of Central Florida; University of Rochester \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Revisiting Self-Supervised Visual Representation Learning (**** - <font color="#dd0000">**CVPR**19</font>) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kolesnikov_Revisiting_Self-Supervised_Visual_Representation_Learning_CVPR_2019_paper.pdf) [[code]](https://github.com/google/revisiting-self-supervised) [[supp]](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Kolesnikov_Revisiting_Self-Supervised_Visual_CVPR_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Alexander Kolesnikov, Xiaohua Zhai, Lucas Beyer \
    > <font color=Gray>**Organization(s)**:</font> Google Brain \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Revisiting

- Local Aggregation for Unsupervised Learning of Visual Embeddings (**LA** - <font color="#dd0000">**ICCV**19</font> Oral) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhuang_Local_Aggregation_for_Unsupervised_Learning_of_Visual_Embeddings_ICCV_2019_paper.pdf) [[code]](https://github.com/neuroailab/LocalAggregation) [[supp]](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Zhuang_Local_Aggregation_for_ICCV_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Chengxu Zhuang, Alex Lin Zhai, Daniel Yamins \
    > <font color=Gray>**Organization(s)**:</font> Stanford University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Clustering

- S4L: Self-Supervised Semi-Supervised Learning (**S4L** - <font color="#dd0000">**ICCV**19</font>) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhai_S4L_Self-Supervised_Semi-Supervised_Learning_ICCV_2019_paper.pdf) [[code]](https://github.com/google-research/s4l) [[supp]](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Zhai_S4L_Self-Supervised_Semi-Supervised_ICCV_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, Lucas Beyer \
    > <font color=Gray>**Organization(s)**:</font> Google Research, Brain Team \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- AVT: Unsupervised Learning of Transformation Equivariant Representations by Autoencoding Variational Transformations (**AVT** - <font color="#dd0000">**ICCV**19</font>) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Qi_AVT_Unsupervised_Learning_of_Transformation_Equivariant_Representations_by_Autoencoding_Variational_ICCV_2019_paper.pdf) [[code]](https://github.com/maple-research-lab/AVT-pytorch) [[supp]](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Qi_AVT_Unsupervised_Learning_ICCV_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Guo-Jun Qi, Liheng Zhang, Chang Wen Chen, Qi Tian \
    > <font color=Gray>**Organization(s)**:</font> G1Laboratory for MAchine Perception and LEarning (MAPLE); Huawei Cloud; Huawei Noah’s Ark Lab;  The Chinese University of Hong Kong at Shenzhen and Peng Cheng Laboratory \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervised Representation Learning via Neighborhood-Relational Encoding (**NRE** - <font color="#dd0000">**ICCV**19</font>) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sabokrou_Self-Supervised_Representation_Learning_via_Neighborhood-Relational_Encoding_ICCV_2019_paper.pdf) [[~~No code~~]]() 
    > <font color=Gray>**Author(s)**:</font> Mohammad Sabokrou, Mohammad Khalooei, Ehsan Adeli \
    > <font color=Gray>**Organization(s)**:</font> Institute for Research in Fundamental Sciences; Amirkabir University of Tech.; Stanford University \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Unsupervised Pre-Training of Image Features on Non-Curated Data (**DeeperCluster** - <font color="#dd0000">**ICCV**19</font>) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Caron_Unsupervised_Pre-Training_of_Image_Features_on_Non-Curated_Data_ICCV_2019_paper.pdf) [[code]](https://github.com/facebookresearch/DeeperCluster) [[supp]](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Caron_Unsupervised_Pre-Training_of_ICCV_2019_supplemental.pdf)
    > <font color=Gray>**Author(s)**:</font> Mathilde Caron, Piotr Bojanowski, Julien Mairal, Armand Joulin \
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research; Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LJK \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Clustering

- Scaling and Benchmarking Self-Supervised Visual Representation Learning (**** - <font color="#dd0000">**ICCV**19</font>) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) [[code]](https://github.com/facebookresearch/fair_self_supervision_benchmark) 
    > <font color=Gray>**Author(s)**:</font> Priya Goyal, Dhruv Mahajan, Abhinav Gupta, Ishan Misra \
    > <font color=Gray>**Organization(s)**:</font> Facebook AI Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

  

## 2018

- Unsupervised Feature Learning via Non-Parametric Instance Discrimination ( **InsDis** - <font color="#dd0000">**CVPR**18</font> Spotlight) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf) [[code]](https://github.com/zhirongw/lemniscate.pytorch)
    > <font color=Gray>**Author(s)**:</font> Zhirong Wu, Yuanjun Xiong, Stella X. Yu, Dahua Lin \
    > <font color=Gray>**Organization(s)**:</font>  UC Berkeley/ICSI; Chinese University of Hong Kong; Amazon Rekognition\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Contrastive learning, single branch

- Improvements to context based self-supervised learning ( **** - <font color="#dd0000">**CVPR**18</font>) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mundhenk_Improvements_to_Context_CVPR_2018_paper.pdf) [[code]](https://gdo-datasci.llnl.gov/selfsupervised/) [[supp]](https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/3991-supp.pdf)
    > <font color=Gray>**Author(s)**:</font> T. Nathan Mundhenk, Daniel Ho, Barry Y. Chen \
    > <font color=Gray>**Organization(s)**:</font> Lawrence Livermore National Laboratory; University of California, Berkeley \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Self-Supervised Feature Learning by Learning to Spot Artifacts ( **** - <font color="#dd0000">**CVPR**18</font>) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jenni_Self-Supervised_Feature_Learning_CVPR_2018_paper.pdf) [[code]](https://github.com/sjenni/LearningToSpotArtifacts)
    > <font color=Gray>**Author(s)**:</font> Simon Jenni, Paolo Favaro \
    > <font color=Gray>**Organization(s)**:</font> University of Bern \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Boosting Self-Supervised Learning via Knowledge Transfer ( **** - <font color="#dd0000">**CVPR**18</font>) [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Noroozi_Boosting_Self-Supervised_Learning_CVPR_2018_paper.pdf) [[~~No code~~]]() 
    > <font color=Gray>**Author(s)**:</font> Mehdi Noroozi, Ananth Vinjimoor, Paolo Favaro, Hamed Pirsiavash \
    > <font color=Gray>**Organization(s)**:</font> University of Bern; University of Maryland, Baltimore County \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Deep Clustering for Unsupervised Learning of Visual Features ( **DeepCluster** - <font color="#dd0000">**ECCV**18</font>) [[paper]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf) [[code]](https://github.com/facebookresearch/deepcluster) 
    > <font color=Gray>**Author(s)**:</font> Caron, Mathilde and Bojanowski, Piotr and Joulin, Armand and Douze, Matthijs \
    > <font color=Gray>**Organization(s)**:</font>  Facebook AI Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 


## 2017

- Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction (**Split-Brain Auto** - <font color="#dd0000">**CVPR**17</font> ) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Split-Brain_Autoencoders_Unsupervised_CVPR_2017_paper.pdf) [[code]](https://github.com/richzhang/splitbrainauto)
    > <font color=Gray>**Author(s)**:</font> Richard Zhang, Phillip Isola, Alexei A. Efros\
    > <font color=Gray>**Organization(s)**:</font>  Berkeley AI Research (BAIR) Laboratory, University of California, Berkeley\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

-  Representation Learning by Learning to Count (**** - <font color="#dd0000">**ICCV**17</font> ) [[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Noroozi_Representation_Learning_by_ICCV_2017_paper.pdf) [[unofficial code]](https://github.com/clvrai/Representation-Learning-by-Learning-to-Count)
    > <font color=Gray>**Author(s)**:</font> Mehdi Noroozi, Hamed Pirsiavash, Paolo Favaro \
    > <font color=Gray>**Organization(s)**:</font>  University of Bern; University of Maryland, Baltimore County \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

-  Transitive Invariance for Self-supervised Visual Representation Learning (**** - <font color="#dd0000">**ICCV**17</font> ) [[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Transitive_Invariance_for_ICCV_2017_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Xiaolong Wang, Kaiming He, Abhinav Gupta \
    > <font color=Gray>**Organization(s)**:</font>  Carnegie Mellon University; Facebook AI Research \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font>

-  Multi-task Self-Supervised Visual Learning (**** - <font color="#dd0000">**ICCV**17</font> ) [[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Doersch_Multi-Task_Self-Supervised_Visual_ICCV_2017_paper.pdf) [[code]](https://github.com/deepmind/multiself)
    > <font color=Gray>**Author(s)**:</font> Carl Doersch, Andrew Zisserman \
    > <font color=Gray>**Organization(s)**:</font>  DeepMind; VGG, Department of Engineering Science, University of Oxford \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

## 2016

- Context Encoders: Feature Learning by Inpainting (**Inpainting** - <font color="#dd0000">**CVPR**16</font> ) [[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf) [[unofficial code]](https://github.com/BoyuanJiang/context_encoder_pytorch)
    > <font color=Gray>**Author(s)**:</font> Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros \
    > <font color=Gray>**Organization(s)**:</font> University of California, Berkeley \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

- Unsupervised Learning of Discriminative Attributes and Visual Representations (**** - <font color="#dd0000">**CVPR**16</font> ) [[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Huang_Unsupervised_Learning_of_CVPR_2016_paper.pdf) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Chen Huang, Chen Change Loy, Xiaoou Tang \
    > <font color=Gray>**Organization(s)**:</font>  The Chinese University of Hong Kong; SenseTime Group Limited; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles (**Jigsaw** - <font color="#dd0000">**ECCV**16</font> ) [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46466-4_5) [[unofficial code]](https://github.com/bbrattoli/JigsawPuzzlePytorch) [[arxiv]](https://arxiv.org/abs/1603.09246)
    > <font color=Gray>**Author(s)**:</font> Mehdi Noroozi, Paolo Favaro\
    > <font color=Gray>**Organization(s)**:</font> Institute for Informatiks, University of Bern \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

- Unsupervised Visual Representation Learning
by Graph-based Consistent Constraints (**** - <font color="#dd0000">**ECCV**16</font> ) [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_41) [[~~No code~~]]()
    > <font color=Gray>**Author(s)**:</font> Dong Li, Wei-Chih Hung, Jia-Bin Huang, Shengjin Wang, Narendra Ahuja; Ming-Hsuan Yang \
    > <font color=Gray>**Organization(s)**:</font> Tsinghua University; University of California, Merced; University of Illinois, Urbana-Champaign \
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Colorful Image Colorization (**** - <font color="#dd0000">**ECCV**16</font> ) [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40) [[code]](https://github.com/richzhang/colorization)
    > <font color=Gray>**Author(s)**:</font> Richard Zhang, Phillip Isola, Alexei A. Efros \
    > <font color=Gray>**Organization(s)**:</font> University of California, Berkeley\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

## 2015

- Unsupervised Visual Representation Learning by Context Prediction (**Context** - <font color="#dd0000">**ICCV**15</font> ) [[paper]](https://openaccess.thecvf.com/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf) [[code]](https://github.com/cdoersch/deepcontext)
    > <font color=Gray>**Author(s)**:</font> Carl Doersch, Abhinav Gupta, Alexei A. Efros\
    > <font color=Gray>**Organization(s)**:</font>  Institute for Informatiks, University of Bern\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task

- Learning to See by Moving (**** - <font color="#dd0000">**ICCV**15</font> ) [[paper]](https://openaccess.thecvf.com/content_iccv_2015/papers/Agrawal_Learning_to_See_ICCV_2015_paper.pdf) [[code]](https://github.com/pulkitag/learning-to-see-by-moving)
    > <font color=Gray>**Author(s)**:</font> Pulkit Agrawal, Joao Carreira, Jitendra Malik \
    > <font color=Gray>**Organization(s)**:</font>  UC Berkeley\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

- Unsupervised Learning of Visual Representations using Videos (**** - <font color="#dd0000">**ICCV**15</font> ) [[paper]](https://openaccess.thecvf.com/content_iccv_2015/papers/Wang_Unsupervised_Learning_of_ICCV_2015_paper.pdf) [[unofficial code]](https://github.com/coreylynch/unsupervised-triplet-embedding)
    > <font color=Gray>**Author(s)**:</font> Xiaolong Wang, Abhinav Gupta \
    > <font color=Gray>**Organization(s)**:</font>  Robotics Institute, Carnegie Mellon University\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> 

## 2014

- Discriminative Unsupervised Feature Learning with Convolutional Neural Networks (**ExemplarCNN** - <font color="#dd0000">**NeurIPS**14</font> ) [[paper]]() [[unofficial code]](https://github.com/yihui-he/Exemplar-CNN)
    > <font color=Gray>**Author(s)**:</font> Alexey Dosovitskiy, Jost Tobias Springenberg, Martin A. Riedmiller, Thomas Brox\
    > <font color=Gray>**Organization(s)**:</font>  Computer Science Department, University of Freiburg\
    > <font color=Gray>**Description**:</font>   \
    > <font color=Gray>**Tags**:</font> Pretext task, instance discrimination


***

UVRL is also closely related to some other research directions, such as **Unsupervised Person (Object or Vehicle) Re-Identification**, **Unsupervised Semantic Segmentation**, **Unsupervised Domain Adaptation**, **Knowledge Distillation**, **Deep Clustering**, **Unsupervised Metric (Embedding) Learning**, **Semi-supervised Learning**, **Novel Categories (Classes) Discovery**, etc.

### **Some Influential Repositories**
- awesome-self-supervised-learning (star 5.0k)  [[link]](https://github.com/jason718/awesome-self-supervised-learning)

- Awesome-Knowledge-Distillation (star 1.8k) [[link]](https://github.com/FLHonker/Awesome-Knowledge-Distillation)

- DeepClustering (star 1.8k) [[link]](https://github.com/zhoushengisnoob/DeepClustering)

- awesome-metric-learning [[link]](https://github.com/qdrant/awesome-metric-learning)

- Awesome-Unsupervised-Person-Re-identification [[link]](https://github.com/Yimin-Liu/Awesome-Unsupervised-Person-Re-identification)

- Awesome-Novel-Class-Discovery [[link]](https://github.com/JosephKJ/Awesome-Novel-Class-Discovery)

***

Thanks for the support of Prof. [Yu Zhou](https://people.ucas.ac.cn/~yuzhou).




