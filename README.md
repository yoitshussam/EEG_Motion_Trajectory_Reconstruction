# Hand Kinematic Reconstruction using EEG signals
 Used Datasets: [WAY-EEG-GAL](https://springernature.figshare.com/collections/WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376), and [FULL BODY IN UNCONSTRAINED MOTION](https://figshare.com/articles/dataset/EEG_Data/5616109?backTo=/collections/Full_body_mobile_brain-body_imaging_data_during_unconstrained_locomotion_on_stairs_ramps_and_level_ground/3934264)

Used Models: CNN+LSTM, cGAN, CAE
## Structure of project:
WAY-EEG-GAL folder:
- import.ipynb \
The code blocks in this file are for importing the EEG and kinematic files, as well as filtering and applying ICA to the EEG

- CNN+LSTM_training.py
- GAN_training.py
- CAE_training.py
These are the training scripts for each different model. They handle normalization as well as windowing, training, and plotting. They are largely the same with the difference being the models and the training loop. 
It's possible to merge them all together and have the models in a seperate file to import as a class

\
FULL BODY folder:
- import.ipynb
The code blocks in this file handle importing the EEG, EOG, and kinematic data. As well as upsampling the kinematic data to match the EEG

