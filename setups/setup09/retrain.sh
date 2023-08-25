rm -rf ./log
rm train.log
rm model_checkpoint_*

bsub -J "setup09" -n 16 -gpu "num=1" -q gpu_a100 -o train.log python /nrs/funke/rhoadesj/xray-challenge-entry/setups/setup09/train.py