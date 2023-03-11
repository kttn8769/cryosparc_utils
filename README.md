# cryosparc_utils

## Install
```
cd <a directory you like>
git clone https://github.com/kttn8769/cryosparc_utils.git
```

## Update
```
cd <cryosparc_utils directory>
git pull
```

## How to use scripts
The scripts in cryosparc_utils use the python environment and modules of the cryoSPARC worker.

1. Activate the cryoSPARC worker's python environment.

```
source <path to the cryosparc_worker directory>/deps/anaconda/bin/activate
conda activate cryosparc_worker_env
```

2. Set PYTHONPATH

```
export PYTHONPATH=<path to the cryosparc_worker>
```

3. Use scripts

```
python <path to the cryosparc_utils directory>/scripts/<script you want to use> --help
```

(Don't forget to type "python" first.)
