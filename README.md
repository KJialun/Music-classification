# Document description
- Basic of signal processing demonstrate how do we extract features from audio .wav file by using MFCC 
- Model_trainning_MFCC contains the code about data preprocessing by using MFCC and different network trainning 
> Dataset can be access from <https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification>\\
> > First: run datapreprocessing.ipynb to generate dara_10.json file 

> > Second: run Genre_classifier_NN.ipynb to train Neuron Network (NN) model  

> > Third: run EnhancedNN_MusicClassification.ipynb to train the new NN model with Dropouts and L2-Norm (regularizer) to fix overfitting

> > Fourth: run CNN_MusicClassification.ipynb to train the Convolutional Neural Network (CNN) --> change (Gio  CNN)

> > Fifth: run *.ipynb to train the Long Short-Term Memory Recursive Neural Network (LSTM-RNN) --> change (Gio  LSTM)
# Below you can find a step-by-step guide to set up your environment.
- Install Anaconda: Download and install Anaconda from the link below (choose the right version for your OS): <https://www.anaconda.com/download/>
- Open anaconda command prompt
- For data preprocessing we will use librosa library ,Create new anaconda environment named "preprocessing"  : 
```
conda create --name preprocessing 
``` 
- Enter the newly created environment:
``` 
conda activate preprocessing
``` 
- Install required packages by sequentially executing:
``` 
conda install -c conda-forge librosa
conda install -c anaconda ipykernel
``` 

- For model training , we will use tensorflow library, Create new anaconda environment named "tensorflow",and download library below : 
``` 
conda create --name tensorflow
conda activate tensorflow
conda install -c anaconda scikit-learn
conda install -c conda-forge tensorflow
conda install -c anaconda ipykernel
``` 

- Open Vscode and select the environment you create .
