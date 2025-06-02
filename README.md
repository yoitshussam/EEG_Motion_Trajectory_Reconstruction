## Hand Kinematic Reconstruction using EEG signals (WAY-EEG-GAL DATASET)
### -import_preprocess.py
contains the code for importing all the eeg and kinematic data as well as applying ICA to them and outputting a file in raw MNE .fif format    
The code assumes that you have downloaded the dataset and extracted the HS_P1_S1.mat—HS_P12_S9.mat (108 files) into a single folder  
Each file contains all data in a single lifting series, in continuous format


https://drive.google.com/file/d/115VUReHSuEn-ICRZ9759aiiCBjQZ-PGq/view?usp=sharing
https://drive.google.com/file/d/1V4ZKLZhdnnM4GoHeaTfmDDRBe5TErqrO/view?usp=sharing

these are the cleaned EEG.fif and kin_data.npy files so you don't have to download the entire dataset to import and preprocess
### -training_and_plot.py 
contains everything else. First it loads the cleaned EEG file as well as the kinematic data and performs a sliding windowing process with a specified lag. Afterwards the training, validation, testing split is done  
and the Normalization is performed and then the data is turned into Tensors and Dataloaders for PyTorch. Finally the training of the model, and the plotting of the results with the PCC score  
command line arguments are:   

#1: participant number  
#2: sliding window shift sample size  
#3: model architecture (1) PreMovNet CNNLSTM, and (2) 2D CNN   

so .\training_and_plot.py 3 10 1 would mean participant #3 shift of 10 samples, and PreMovNet model

