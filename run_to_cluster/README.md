# Reproducibility

## Run on cluster

The code was run on a cluster with 1 GPU for computational reason (with runai) and docker (for more information check this [tutorial](https://github.com/epfl-ml4ed/runai-tutorial)

Below are the steps to run the code:
- Starting docker (with commande line or directly on the app).
```Docker
docker start
```
- Build a Docker image with the tag image_name from the current directory (indicated by the . at the end).
```Docker
docker build -t image_name .
```
- Login to runai
```bash
runai login (only the first time)
```
- Login to the registry
```Docker
docker login [registry]
```
- Tag your image to the registry
```Docker
- docker tag image_name [registry]/image_name
```
- Push the image
```Docker
docker push [registry]/image_name
```

**Submit the job**
Depends on which python file one wants to run, the command line in terminal is different:
  - Multi-class classification: use [train_multiclass_clf_CV.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiclass_clf_CV.py)

```bash
runai submit --name NAME_JOB -p [id_runai] -i [registry]/image_name --cpu-limit 1 --gpu 1 -- python train_multiclass_clf_CV.py --batch_size 8 --epochs 10 --n_splits 5
```
  - Multiple Binary Classifiers: use [train_multiple_bin_clf_CV.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiple_bin_clf_CV.py)

```bash
runai submit --name NAME_JOB -p [id_runai] -i [registry]/image_name --cpu-limit 1 --gpu 1 -- python train_multiple_bin_clf_CV.py --batch_size 8 --epochs 10 --epochs_eval 3 --n_splits 5 --topN 7
```
  - Multiple Binary Classifiers with Downsampling: use [train_multiple_bin_clf_CV_downsampled.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiple_bin_clf_CV_downsampled.py)

```bash
runai submit --name NAME_JOB -p [id_runai] -i [registry]/image_name --cpu-limit 1 --gpu 1 -- python train_multiple_bin_clf_CV_downsampled.py --batch_size 8 --epochs 5 --epochs_eval 3 --n_splits 5 --topN 7 --cut_downsampling_train 600 --cut_downsampling_test 200
```
  - Leanrning Curves Approach: use [train_multiple_bin_clf_CV_downsampled_LC.py](https://github.com/Maximelel/SP_in_ML4ED/blob/main/run_to_cluster/train_multiple_bin_clf_CV_downsampled_LC.py)

```bash
runai submit --name NAME_JOB -p [id_runai] -i [registry]/image_name --cpu-limit 1 --gpu 1 -- python train_multiple_bin_clf_CV_downsampled_LC.py --batch_size 8 --epochs 2 --N_shuffle_total 5 --topN 7 --cut_downsampling 600
```

## Other commands after launching the run

### Check the status of the job
```bash
runai describe job NAME_JOB -p [id_runai]
```
### Check the logs
```bash
 kubectl logs NAME_JOB -n runai-[id_runai]
```
### Delete a job
```bash
runai delete job -p [id_runai] NAME_JOB
```
### See the list of jobs
```bash
runai list jobs -p [id_runai]
```

