# Readme

## conda

[conda env](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments)

`conda activate rl-test`
`conda env create -f environment.yml`
`conda env export --from-history > environment.yml`


[update conda env from yml](https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file)

```shell
conda activate myenv
conda env update --file local.yml --prune
```a

or 

```shell
conda env update --name myenv --file local.yml --prune
```

## stable baselines

### install

#### requires pytorch

[install stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

https://pytorch.org/get-started/locally/

cuda 11.3 for rtx 3070

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

problems with sm_86

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly` could be helpful

restart fixed the problem

#### install via pip

https://stable-baselines3.readthedocs.io/en/master/guide/install.html

`pip install stable-baselines3[extra]`	

### highway_env

https://github.com/eleurent/highway-env

## HER

works https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#hindsight-experience-replay-her
