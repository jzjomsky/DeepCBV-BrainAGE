# Brain Age Gap Estimation (BrainAGE) with T1w MRI and DeepCBV

Jordan Jomsky, Zongyu Li, Kay C Igwe, Yiren Zhang, Max Lashley, Tal Nuriel, Andrew Laine, Scott A Small, Jia Guo, for the Frontotemporal Lobar Degeneration Neuroimaging Initiative and for the Alzheimer’s Disease Neuroimaging Initiative , Enhancing brain age estimation with structural MRI and synthesized cerebral blood volume maps, Brain Communications, Volume 8, Issue 5, 2026, fcag283, https://doi.org/10.1093/braincomms/fcag283

This repository contains a VGG-style 3D CNN architecture for Brain Age estimation from T1-weighted MRI and AI-Synthesized Cerebral Blood Volume data and two pretrained weight files (BrainAGE_T1_Model_Weights.pkl, T1-only model and BrainAGE_DeepCBV_Model_Weights.pkl, DeepCBV-only model) found within the BrainAGE Model folder. Within the DeepCBV Model folder, there is a separate README with the instructions for generating DeepCBV with the requisite code and model weights.
 
The multimodal approach (T1 + DeepCBV) improves predictive performance (MAE ≈ 3.95 years, R² ≈ 0.943 on held-out test set) vs. unimodal T1 or DeepCBV models. The project and manuscript were developed in Jia Guo’s Lab at Columbia University. Contact: Jia Guo, jg3400@columbia.edu. 
 
If you use this code or models, please cite the official manuscript: 
@article{10.1093/braincomms/fcag283,
    author = {Jomsky, Jordan and Li, Zongyu and Igwe, Kay C and Zhang, Yiren and Lashley, Max and Nuriel, Tal and Laine, Andrew and Small, Scott A and Guo, Jia and for the Frontotemporal Lobar Degeneration Neuroimaging Initiative and for the Alzheimer’s Disease Neuroimaging Initiative },
    title = {Enhancing brain age estimation with structural MRI and synthesized cerebral blood volume maps},
    journal = {Brain Communications},
    volume = {8},
    number = {5},
    pages = {fcag283},
    year = {2026},
    month = {10},
    issn = {2632-1297},
    doi = {10.1093/braincomms/fcag283},
    url = {https://doi.org/10.1093/braincomms/fcag283},
    eprint = {https://academic.oup.com/braincomms/article-pdf/8/5/fcag283/69807180/fcag283.pdf},
}
 
This study used and aggregated many public neuroimaging datasets (ADNI, AIBL, OASIS, IXI, PPMI, BGSP, SLIM, DLBS, SALD, CoRR, SchizConnect, and FTLDNI). See the manuscript for the complete acknowledgements and investigator lists.  
