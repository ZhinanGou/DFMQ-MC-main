# DFMQ-MC:Dynamic Fusion Based on Modality Quality and Cross-Modal Semantic Consistency for Multimodal Intent Recognition

## 1、Introduction

The core challenge in multimodal intent recognition is effectively integrating complementary information from text, audio, and video modalities to infer a user’s true intent. However, existing methods typically lack explicit awareness of modality quality, resulting in an inability to dynamically assign fusion weights or resolve cross-modal semantic conflicts, leading to substantial performance degradation under low-quality or heterogeneous modal conditions. To address these limitations, this paper proposes DFMQ-MC, which employs modality quality-aware dynamic fusion and cross-modal semantic consistency constraints to guide the learning of quality-adaptive, semantically aligned discriminative cross-modal features, thereby enabling more robust multimodal intent recognition.

## 2、Run on GPU:

Model runs on GPU by default with cuda:0. This experiment was performed on a 4090 GPU.

## 3、Dependencies：

We use anaconda to create python environment:

```
cd DFMQ-MC
conda create --name dfmq-mc python=3.9
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3
```

Install all required libraries:

```
pip install -r requirements.txt
```

## 4、Usage：

The data can be downloaded through the following links:

```
https://drive.google.com/file/d/16f1SOamp_hRuqRqH37eophnSQr1Dl2_w/view?usp=sharing # MIntRec
https://drive.google.com/file/d/1Pn-Tqok36goVdJtuxzx4fsEP0aVjeKRb/view?usp=sharing # MELD
```

You can evaluate the performance of our proposed DFMQ-MC on [MIntRec](https://dl.acm.org/doi/pdf/10.1145/3503161.3547906) and [MELD-DA](https://aclanthology.org/2020.acl-main.402.pdf) by using the following commands:

- MIntRec

```
sh examples/run_mcwp_mintrec.sh
```

- MELD-DA

```
sh examples/run_mcwp_meld_da.sh
```

You can change the parameters in the **configs** folder. The default parameter is the best parameter on two datasets

## 5、Model

The overview model architecture:

<img width="1464" height="549" alt="image" src="https://github.com/user-attachments/assets/3e6c8226-3275-48ed-a635-98940edcec31" />


## 6、Experimental Results

<img width="1479" height="564" alt="image" src="https://github.com/user-attachments/assets/4f6361b2-1bb6-461f-b250-79322eb04854" />


## 7、Citation

If you are insterested in this work, and want to use the codes or results in this repository, please **star** this repository and **cite** by:

```

```

## 8、Acknowledgments

Some of the codes in this repo are adapted from [MIntRec](https://github.com/thuiar/MIntRec/tree/main), and we are greatly thankful.

If you have any questions, please open issues and illustrate your problems as detailed as possible.

