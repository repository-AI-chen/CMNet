# CMNet
Multi-scale feature compression for machine intelligence aims to remove redundancies while minimizing the performance degradation of machine vision tasks. However, existing multi-scale feature compression methods remove cross-scale redundancies between different scale features, but they fail to efficiently eliminate intra-scale redundancies in a single-scale feature. To efficiently compress multi-scale features, a Collaborative Mining Network (CMNet) is proposed, which simultaneously exploits intra-scale and cross-scale correlations to eliminate task-irrelevant redundancies. To effectively remove intra-scale redundancies, an intra-scale redundancy-aware module is proposed, in which a swin transformer block and a multi-receptive-field local perception block are designed to capture global and diverse local correlations in a single-scale feature, and an attention-driven selective aggregation block is developed to extract critical information relevant to machine vision tasks. In addition, to efficiently remove cross-scale redundancies, a context-guided adaptive fusion module is proposed for dynamically mining both spatial and channel correlations between different scale features. Experimental results demonstrate that our proposed CMNet achieves superior coding efficiency, and outperforms other state-of-the-art feature compression methods.
# Training and Testing
## (1) Training Stage
python train.py
## (2) Finetuning Stage
python train_finetune.py
## (3) Testing Stage
python test_and_results_all_in_folder_seg.py
# Requirements
(1) torch==2.4.1  
(2) compressai==1.2.6  
(3) detectron2==0.6  
(4) tensorboard=2.14.0  
(5) numpy==1.22.4  
