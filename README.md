# Hand Kinematic Reconstruction using EEG signals
 Used Datasets: [WAY-EEG-GAL](https://springernature.figshare.com/collections/WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376), and [FULL BODY IN UNCONSTRAINED MOTION](https://figshare.com/articles/dataset/EEG_Data/5616109?backTo=/collections/Full_body_mobile_brain-body_imaging_data_during_unconstrained_locomotion_on_stairs_ramps_and_level_ground/3934264)

Used Models: CNN+LSTM, cGAN, CAE

## The scripts of each dataset:
WAY-EEG-GAL folder:
- import.ipynb \
The code blocks in this file are for importing the EEG and kinematic files, as well as filtering and applying ICA to the EEG.
To run the data processing script, download the original dataset which is seperated into 12 folders P1,P2, etc. for each participant.
Each folder has
-HS_P1_S1.mat—HS_P12_S9.mat and
-WS_P1_S1.mat—WS_P12_S9.mat files
they are the same data but HS is given in the format we want (continuous). We also need the AllLifts files which we use to seperate individual trials and extract only the motion events without the rest inbetween.
Run the script in the same directory as the folders of dataset.

- CNN+LSTM_training.py
- GAN_training.py
- CAE_training.py
These are the training scripts for each different model. They handle normalization as well as windowing, training, and plotting. They are largely the same with the difference being the models and the training loop. 
It's possible to merge them all together and have the models in a seperate file to import as a class

\

FULL BODY folder:
- import.ipynb
The code blocks in this file handle importing the EEG, EOG, and kinematic data. As well as upsampling the kinematic data to match the EEG.
To run the script, download the dataset and extract it into 'data' folder into the same directory and run. 

- fullbody_CNN+LSTM_training.py
- fullbody_GAN_training.py
- fullbody_CAE_training.py
