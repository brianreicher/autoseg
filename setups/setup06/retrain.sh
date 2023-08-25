rm -rf ./log
rm train.log

bsub -J "setup06" -n 16 -gpu "num=1" -q gpu_a100 -o train.log python /nrs/funke/rhoadesj/xray-challenge-entry/setups/setup06/train.py