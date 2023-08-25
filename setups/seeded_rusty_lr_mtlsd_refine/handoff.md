# Janelia Handoff


## Data

### SOTA RR

- All STELARRR setup data with `./sota_rr_data`

    Contains . . .
    - Final checkpoint (currently 220k) affs, frags, segmentation 
    - Initial checkpoint (10k) affs, frags, segmentation
    - All model checkpoints

### No-Rinse

- All non-rinse setup data with `./no_rinse_data`

    Contains . . .
    - Final checkpoint (currently 220k) affs, frags, segmentation 
    - Initial checkpoint (10k) affs, frags, segmentation
    - All model checkpoints


## Training

- For STELARRR training: `python ./train.py`

- For no-rinse training: `python ./rinseless_train.py`

## Validation Segmentation (used in post-processing pipeline)

- Rusty Mutex Watershed:
    - Command: `python ./rusty_mws_eval_seg.py`
    - References driver function `get_pred_segmentation()` in `rusty_mws/rusty_segment_mws.py`
    - Running alone would write to `./validation.zarr` (affs, frags, seg)

- Heiarchical Agglomeration Segmentation
    - Command : `bash ./heiarchical_agglom/segment.sh`
    - References individual segmentation steps and config jsons 02-05
    - Running alone would write to `./validation.zarr` (relies on pre-written affs, will write frags and seg)


## Skeleton-Corrected Rusty Segmentation (used in training pipeline)

- Command: `python ./rusty_skel_correct_seg.py`
- References driver function `get_corrected_segmentation()` in `rusty_mws/rusty_segment_mws.py`
- Running alone would write to `./raw_predictions.zarr` (affs, frags, seg, masks)

## Validation

- Command: `python ./get_eval.py`
- Calls `get_validation_segmentation()` from `./rusty_mws_eval_seg.py`
- Will generate affs based on the availible specified checkpoint in the `./` directory, saved into `./validation.zarr`
- Will generate seeded rusty MWS segmentation, saved into `./validation.zarr`
- Reads segmentation from `./validation.zarr`

## Re-Run MWS Validation Segmentation from Pre-Generated Fragments & New Parameters

- If you have zarr-saved frags and the name of a MongoDB collection that these frag nodes/edges are saved to, you can re-segment on the same frags
- Open `./rusty_mws/rusty_segment_mws.py` and in the `get_pred_segmentation()` function, make sure to commment out first two function calls in the pipeline
- Set the `sample_name` for `global_mutex_watershed_on_supervoxels` and `extract_segmentation` to your saved mongo collection name (i.e. "htem39454661040933637")
- Navigate to `./rusty_mws_eval_seg.py` and comment out the affinities prediction call
- Enter whatever deisred parameters you choose in this script for `adj_bias` and `lr_bias` and run `python ./rusty_mws_eval_seg.py`

## Run an Evolutionary Hyperparameter Search

- Can run an evolutionary hyperparmeter search with `python ./optimize_mutex_weights_evo.py`
- Same conditions as above need to apply - can tweak population and generation size at bottom of script, and enter the MongoDB collection name as a parameter for the evolutionary call as well
- BUG EXISTS: need to figure out why every successive segmentation just re-uses the same LUT as the first member of the population
- Otherwise, it works and it is set to evolved based on an average of `VOI_split` and `VOI_merge` scores, which are just not changing from generation to generation because I think it's reading from a stale LUT