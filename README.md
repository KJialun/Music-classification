# Music-classification
> Below you can find a step-by-step guide to set up your environment.
- Install Anaconda: Download and install Anaconda from the link below (choose the right version for your OS): <https://www.anaconda.com/download/>
- Open anaconda command prompt
- Create new anaconda environment named "preprocessing" you will use this environment to do datapreprocessing : 
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

- Create new anaconda environment named "tensorflow" , you will use this environment to do trainning : 
``` 
conda create --name tensorflow
conda activate tensorflow
conda install -c anaconda scikit-learn
conda install -c conda-forge tensorflow
conda install -c anaconda ipykernel
``` 

- Open Vscode and select the environment you create .