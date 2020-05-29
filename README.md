# pionsDL
Git repository for pions study with Graph Neural Networks.

Create an anaconda environment first:

`conda create --name pienv python=3.7`


Activate the environment with the command:

`conda activate pienv`


Install packages in `requirements.txt` with the command:

`pip install -r requirements.txt`

If ROOT is needed to visualize plots in `.root` file, get a CMSSW release and activate CMSSW environment (i.e.):

```
scram list
cmsrel CMSSW_11_1_0_pre8
cd CMSSW_11_1_0_pre8/src/
cmsenv
```

To deactivate the environment:

`conda deactivate`