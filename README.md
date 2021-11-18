# Readme

## conda

[conda env](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments)

`conda activate rl-test`
`conda env create -f environment.yml`
`conda env export --from-history > environment.yml`

## stable baselines

### install

#### requires pytorch

[install stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

https://pytorch.org/get-started/locally/

cuda 11.3 for rtx 3070

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

#### install via pip

https://stable-baselines3.readthedocs.io/en/master/guide/install.html

`pip install stable-baselines3[extra]`	

