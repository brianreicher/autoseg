rm segment.log

bsub -J "segment16" -n 24 -gpu "num=1" -q gpu_a100 -o segment.log python /nrs/funke/rhoadesj/xray-challenge-entry/setups/setup16/get_segmentation.py

# bsub -J "segment16" -n 24 -gpu "num=1" -q gpu_tesla -o segment.log python /nrs/funke/rhoadesj/xray-challenge-entry/setups/setup16/get_segmentation.py

tail -F segment.log