# Brain Age Gap Estimation (BrainAGE) with T1w MRI and AICBV

_Enhancing Brain Age Estimation with a Multimodal 3D CNN Approach Combining Structural MRI and AI-Synthesized Cerebral Blood Volume Measures
Jordan Jomsky, Kay C. Igwe, Zongyu Li, Yiren Zhang, Max Lashley, Tal Nuriel, Andrew Laine, Jia Guo, et al.￼_ 
 
**Official preprint**: https://arxiv.org/abs/2412.01865 
 
This repository contains: 
	•	A VGG-style 3D CNN architecture for Brain Age estimation from T1-weighted MRI and AI-Synthesized Cerebral Blood Volume data.  
	•	Two pretrained weight files (examples): 
	  •	BrainAGE_T1_Model_Weights.pkl — T1-only model 
	  •	BrainAGE_AICBV_Model_Weights.pkl — AICBV-enhanced model 
 
The multimodal approach (T1 + AICBV) improves predictive performance (MAE ≈ 3.95 years, R² ≈ 0.943 on held-out test set) vs. unimodal T1 or AICBV models . The project and manuscript were developed in Jia Guo’s Lab at Columbia University. Contact: Jia Guo, jg3400@columbia.edu. 
 
If you use this code or models, please cite the arXiv preprint: 
Jomsky J, Igwe KC, Li Z, Zhang Y, Lashley M, Nuriel T, Laine A, Guo J, et al. 
Enhancing Brain Age Estimation with a Multimodal 3D CNN Approach Combining Structural MRI and AI-Synthesized Cerebral Blood Volume Measures. 
Preprint. arXiv:2412.01865. 2024. 
https://arxiv.org/abs/2412.01865 
 
This study used and aggregated many public neuroimaging datasets (ADNI, AIBL, OASIS, IXI, PPMI, BGSP, SLIM, DLBS, SALD, CoRR, SchizConnect, and FTLDNI). See the manuscript for the complete acknowledgements and investigator lists.  
