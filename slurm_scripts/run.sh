#!/bin/bash
#SBATCH  --gres=gpu:v100l:1
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=4
#SBATCH  --mem-per-cpu=32000
#SBATCH  --account=ADD_ACCOUNT_HERE
#SBATCH  --time=0-23:00

source /PATH/TO/ENVIRONMENT/bin/activate
python /PATH/TO/REPOSITORY/Channel_Search/main.py --config='./configs/config_gamma0p9.yaml' --gamma=0.8 --optimization_algorithm='SA' --post_fix=1