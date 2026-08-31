# SmokeMon

This is the code repository for the paper: ['SmokeMon: Unobtrusive Extraction of Smoking Topography Using Wearable Energy-Efficient Thermal'](https://dl.acm.org/doi/abs/10.1145/3569460)

SmokeMon is a wearable device that is capable of detecting smoking sessions reliably. We found that we can reliably, with higher accuracy than previous systems, detect smoking events using a smart wearable neck-worn device that relies on capturing heat signatures from thermal sensors along with its accompanying algorithm. SmokeMon detects smoking puffs using a single high-information-sensing modality that is comfortable to wear and low power (it lasts for a full waking day). Unlike regular color cameras, SmokeMon can capture challenging smoking cases without causing privacy concerns or discomfort to the wearer or bystanders. SmokeMon extracts smoking topography without requiring contact with the cigarette or interfering with smoking habits. SmokeMon measures smoking in much greater detail; going beyond just pack-years or cigarettes-per-day; and can do so automatically without bothering the user with questions.


## Dataset

19 participants were recruited, 115 smoking sessions were generated and examined in both controlled and free-living experiments. The dataset alongside the ground truth labels is available for researchers.

## Smoking Detection Pipeline
### Gesture Detection

- [Download](https://drive.google.com/file/d/181u2wHg3av6dJ6HC1y90xeGCeEm61s7u/view?usp=sharing) the pretrained model for MobileNet V2 and put it under `saved/pretrained`
- Modify json config files (`/configs/mobilenet_v2_3d/*.josn`) with the right dataset directory (data_dir).
- Run the experiment bash script `./exp_scripts/train.sh` 


### Session Detection


# SmokeMon Data Summary

Source data lives in SharePoint at [SmokeMon/Data]((https://nuwildcat.sharepoint.com/:f:/r/sites/FSM-HLSP/Shared%20Documents/Grants/SmokeMon/Data?d=w33ad84d9a5cc422eaa8e1e9027a4f0c4&csf=1&web=1&e=oGWFLK)). That folder has **in_lab** and **in_wild** subfolders.

## in-lab

| Paper | Puff Count | Mean volume (ml) | Mean duration (s) | SharePoint Directory |
| :---- | ---------: | ---------------: | ----------------: | :------------------- |
| P1 | 13 | 170.06 | 2.62 | `SmokeMon/Data/in-lab/S1` |
| P2 | 20 | 207.71 | 2.36 | `SmokeMon/Data/in-lab/S2` |
| P3 | 21 | 92.92 | 1.75 | `SmokeMon/Data/in-lab/S3` |
| P4 | 25 | 102.92 | 2.53 | `SmokeMon/Data/in-lab/S4` |
| P5 | 23 | 274.89 | 3.90 | `SmokeMon/Data/in-lab/S5` |
| P6 | 30 | 244.69 | 4.61 | `SmokeMon/Data/in-lab/S6` |
| P7 | 23 | 124.22 | 2.91 | `SmokeMon/Data/in-lab/S7` |
| P8 | 11 | 143.44 | 2.15 | `SmokeMon/Data/in-lab/S8` |
| **Total / mean** | **166** | **170.11** | **2.85** | |

## in-wild

Puff volume is not available in-wild: there was no ground-truth device, since CReSS Pocket was in-lab only.

| Paper | Puff Count | Mean duration (s) | SharePoint Directory |
| :---- | ---------: | ----------------: | :------------------- |
| P9 | 12 | 1.64 | `SmokeMon/Data/in-wild/S9` |
| P10 | 160 | 4.11 | `SmokeMon/Data/in-wild/S10` |
| P11 | 177 | 2.75 | `SmokeMon/Data/in-wild/S11` |
| P12 | 204 | 2.48 | `SmokeMon/Data/in-wild/S12` |
| P13 | 48 | 1.58 | `SmokeMon/Data/in-wild/S13` |
| P14 | 173 | 1.46 | `SmokeMon/Data/in-wild/S14` |
| P15 | 49 | 1.71 | `SmokeMon/Data/in-wild/S15` |
| P16 | 124 | 3.56 | `SmokeMon/Data/in-wild/S16` |
| P17 | 69 | 2.57 | `SmokeMon/Data/in-wild/S17` |
| P18 | 131 | 3.43 | `SmokeMon/Data/in-wild/S18` |
| P19 | 20 | 1.56 | `SmokeMon/Data/in-wild/S19` |
| **Total / mean** | **1167** | **2.44** | |

## Data Spec

| System | Sensor | Resolution | FoV |
| :----- | :----- | :--------- | :-- |
| SmokeMon | MLX90640 | 32 x 24 | 110 x 75 |
| ASHES | MLX (confirm model) | confirm | confirm |
