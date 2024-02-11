# Reproducibility

## Run on cluster

The code was run on a cluster with 1 GPU for computational reason (with runai) and docker (for more information check this [tutorial](https://github.com/epfl-ml4ed/runai-tutorial)

Below are the steps to run the code:
- Connect to docker with docker desktop (docker login or directly on the app).
- docker build -t image_name .
- runai login (only the first time)
- docker login [registry]
- docker tag image_name [registry]/image_name
- docker push [registry]/image_name
### Submit the job
Depends on which code one wants to run:
  - Multi-class classification: use [train_multiclass_clf_CV.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiclass_clf_CV.py)
runai submit --name experiment1-run1 -p [id_runai] -i [registry]/experiment1 --cpu-limit 1 --gpu 1 -- python train_multiclass_clf_CV.py --batch_size 8 --epochs 10 --n_splits 5

  - Multiple Binary Classifiers: use [train_multiple_bin_clf_CV.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiple_bin_clf_CV.py)


  - Multiple Binary Classifiers with Downsampling: use [train_multiple_bin_clf_CV_downsampled.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiple_bin_clf_CV_downsampled.py)
  - Leanrning Curves Approach: use [train_multiple_bin_clf_CV_downsampled_LC.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiple_bin_clf_CV_downsampled_LC.py)
- runai submit --name [NAME] -p [id_runai] -i [registry]/image_name --cpu-limit 1 --gpu 1

# check the status of the job
runai describe job hello1 -p ml4ed-lelievre

# check the logs
 kubectl logs hello1-0-0 -n runai-ml4ed-lelievre

# delete a job
runai delete job -p ml4ed-lelievre hell01

# see the list of jobs
runai list jobs -p ml4ed-lelievre

