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


