# pionsDL
Git repository for pions study with Deep Neural Networks.

First install anaconda on your local/remote machine (accept the Licence Agreement and allow Anaconda to be added to your `PATH`)

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh 
```

Source the `.bashrc` to load the new `PATH` variable (or logout and login back). Verify that Anaconda is correctly installed and linked by opening Python from the command line. Python from Anaconda should be listed as below

```
$ python
Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```


Now that the installation is complete, create an Anaconda environment

`conda create --name pienv python=3.7`


To work with the repository, activate the environment with the command

`conda activate pienv`


Install packages in `requirements.txt` with the command

`pip install -r requirements.txt`


If ROOT is needed to visualize plots in `.root` file, get a CMSSW release and activate CMSSW environment instead of Anaconda environment (i.e.)

```
scram list
cmsrel CMSSW_11_1_0_pre8
cd CMSSW_11_1_0_pre8/src/
cmsenv
```


To deactivate Anaconda environment

`conda deactivate`
