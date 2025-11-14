# Unmasking real-world audio deepfakes: A data-centric approach

This is the official repository for the paper:

> David Combei, Adriana Stan, Dan Oneata, Nicolas Müller, Horia Cucu,  
> ["Unmasking real-world audio deepfakes: A data-centric approach"](https://arxiv.org/abs/2506.09606),  
> accepted at Interspeech 2025, Rotterdam, Netherlands.

---

## Citation

```bibtex
@inproceedings{combei_interspeech25,
  author={Combei, David and Stan, Adriana and Oneata, Dan and Müller, Nicolas and Cucu, Horia},
  booktitle={Proc. of Interspeech},
  title={{Unmasking real-world audio deepfakes: A data-centric approach}},
  year={2025}
}
```
## Disclaimer

The dataset referenced in this paper is shared via publicly available links. We do not own the copyrights to redistribute the original samples directly.
The links included in this repository were valid as of February 2025. Any content removed from social media platforms before this date is not included.
Please note that some links may become unavailable over time due to platform policies regarding AI-generated or manipulated content or just by owners removing the content from the platform.

## SETUP

### 1. Datasets and dependencies

   Download the **scientific datasets** from their original repositories:
  - [ASV19](https://datashare.ed.ac.uk/handle/10283/3336)
     
  - [ASV21](https://www.asvspoof.org/index2021.html)
     
  - [ASV5](https://zenodo.org/records/14498691)
  - [MLAAD](https://deepfake-total.com/mlaad)
  - [TIMIT](https://zenodo.org/records/6560159)
  - [ODSS](https://zenodo.org/records/8370668)
  - [FoR](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset/data)
  
   Download the **real-life** datasets:
  - [ITW](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa)
  - Links for the AI4T data are available in `AI4T dataset` directory. **NOTE**: Each audio file was segmented into 10 seconds chunks for our experiments.

   To install all the dependencies, run:

   ```
   pip install -r requirements.txt
   ```

### 2. Feature Extraction

   In our implementation we used the features extracted from the **pretrained** SSL model [wav2vec2-xls-r-2b](https://huggingface.co/facebook/wav2vec2-xls-r-2b).
   You can extract the averaged pool representation from each of its 48 layers using the following script. Please make sure to change the paths in the script.
    ```
    python wav2vec2-xls-r-2b_all-layers_extractor.py
    ```
    
  The output of this script will consist of 49 `.npy` files, one from each of the model's layers, where layer 0 is the output for convolution block and 1-48 are the transformers. For each audio sample a `1x1920` shaped vector will be created and stored in the order given by the processed_metadata file. In our experiments, the 9th transformer layer performed best (you can check the extended results [here](https://github.com/davidcombei/AI4T/blob/main/Layers_eval.pdf).
  
   For data augmentation (as introduced in Table 2), we used the first 2 algorithms of Rawboost as described in Tak et al. in [RawBoost](https://arxiv.org/abs/2111.04433).  As codecs, we selected AAC and Opus, each with a probability of 0.5. To generate these augmented features, run:
    
   ```
   python wav2vec2-xls-r-2b_withRawboost_extractor.py
   ```
   
   and
   
   ```
   python wav2vec2-xls-r-2b_withCodec_extractor.py
   ```

These scripts will extract the features from layer 9 alone.

   
## EXPERIMENTS
    
All paths required for the experiments are set in ` config.py `. Please modify them accordingly.
    
   ### 1. Baseline deepfake detector: 
   To reproduce this experiment faster, extract the features from all the datasets using the first extraction script. After you identify the best performing layer, use the augmented extractors to extract the augmented ASV19 train+dev features from that layer. 
   
   To evaluate the performance of each layer, run :
```
python baseline_logReg_all_layers.py
```

Output example:

```
Using layer 9...
[asv19_eval ] EER: 0.1  
[asv21      ] EER: 2.3  
[asv5       ] EER: 0.9  
[for        ] EER: 6.6  
[mlaad      ] EER: 12.8  
[odss       ] EER: 16.2  
[timit      ] EER: 5.6  
[itw        ] EER: 3.4  
[ai4trust   ] EER: 27.4
```

In order to run the baseline deepfake detector with the data augmentation, run:

```
python baseline_logReg_augm.py
```

This process has some randomness due to the data augmentation, so results will likely have small differences.

    
 
   ![image](https://github.com/user-attachments/assets/948ea6cd-de00-412d-ac3c-80a7b95f0d13)
    
   We can see that data augmentation improves the results for scientific datasets, in contrast with the behaviour while evaluating on real-world datasets.
   
   ### 2. Dataset mixing
   
   From the seven scientific datasets, we perform dataset mixing to find to most relevant datasets for generalization on real-world datasets. We train the Logistic Regression classifier using all dataset combinations (127) using the non-augmented features. You can run:
      ```
      python train_logReg_iterative.py
      ```
  
  The script will save a log file named `results.txt`. 
  Output example:
  
  ```
  Datasets: ['FoR', 'MLAAD'], ITW_eer:2.85, AI4TRUST_eer:18.37
  Datasets: ['FoR', 'ASV5'], ITW_eer:2.88, AI4TRUST_eer:32.43
  Datasets: ['FoR', 'ASV21'], ITW_eer:2.62, AI4TRUST_eer:43.97
  Datasets: ['TIMIT', 'MLAAD'], ITW_eer:8.45, AI4TRUST_eer:22.63
  Datasets: ['TIMIT', 'ASV5'], ITW_eer:2.95, AI4TRUST_eer:27.09
  Datasets: ['TIMIT', 'ASV21'], ITW_eer:4.36, AI4TRUST_eer:29.93
  Datasets: ['MLAAD', 'ASV5'], ITW_eer:8.35, AI4TRUST_eer:24.78
  Datasets: ['MLAAD', 'ASV21'], ITW_eer:5.67, AI4TRUST_eer:28.54
  ```
  You can check the extended results for this experiment [here](https://github.com/davidcombei/AI4T/blob/main/Dataset_mixing.pdf).
    
   
   ### 3. Data Pruning
   
  After finding the best dataset mixing combination, we move to data pruning strategies for all seven datasets combined (ALL) and best dataset combination from our paper (FoR+ODSS+MLAAD):

  For random pruning you can run:
      ```
      python pruning_random.py
      ```
  
  This script will randomly select samples from all seven training sets combined, using selection percentages ranging from 10% to 90%. For each percentage, the sampling is repeated 3 times with different random seeds.
  NOTE: We reported the average results of 3 random seeds. Results might not look the same in your case.
  
  For cluster based pruning run:
      ```
      python pruning_cluster.py
      ```
  This approach computes a centroid using `NearestCentroid` and ranks the samples based on Euclidean distance. We apply this strategy independently on both real and fake samples from each of the N datasets, and obtain 2 * N sets of samples selected, which we ensemble in the final pruned dataset. It selects either the `cluster-closest` or `cluster-furthest` based on the `order` parameter.
  
  For margin pruning run:
      ```
      python pruning_margin.py
      ```
  In this script, we use the logistic regression's margins over the samples. We remove the closest or the closest and furthest samples with respect to the decision hyperplane, irrespective of the dataset they pertain to. 
  For removing the closest samples use set the `strategy` parameter to `noisy` and in order to remove both closest and furthest, set the parameter to `both`.
  Below you can see the results for the data selection experiments when using ALL datasets, or only the FoR+ODSS+MLAAD:
  
  ![image](https://github.com/user-attachments/assets/34a16acb-cf50-4d01-9dd5-54a8a140bfe8)


  
  ### 4. Post pruning data augmentation:
  The margin pruning script will save a `.txt` file containing the selected samples. You can use this extended list and augment the training data again using Rawboos and codecs. You can then run:
      ```
      python run_logReg_deepfake_detection_WAugm_margin_pruning.py
      ```
  
  This script will train the logistic regression classifier from scratch using the most relevant data and their augmentations, resulting in the best version in our paper, as in the table below.
  NOTE: this process has some randomness due to the augmentation process, so results are likely to be slightly different. Below you can see the results with the data augmentation post pruning:

  ![image](https://github.com/user-attachments/assets/1a59428f-1257-46c1-8556-d4ad75e51f87)


## ACKNOWLEDGEMENTS

This work was co-funded by EU Horizon project AI4TRUST (No. 101070190), and by the Romanian
Ministry of Research, Innovation and Digitization project DLT-
AI SECSPP (id: PN-IV-P6-6.3-SOL-2024-2-0312), and by the
Free State of Bavaria under the DSgenAI project (Grant Nr.:
RMF-SG20-3410-2-18-4)
      
## LICENSE

The code and data is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
