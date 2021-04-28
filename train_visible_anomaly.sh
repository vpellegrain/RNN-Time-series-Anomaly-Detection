#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=50g
#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --output=/gpfs/workdir/pellegrainv/logs/%j.stdout
#SBATCH --error=/gpfs/workdir/pellegrainv/logs/%j.stderr
#SBATCH --job-name=preproc-fairseq

module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199
module load sox/14.4.2/gcc-9.2.0 
module load gcc/9.2.0/gcc-4.8.5 
module load gcc/8.4.0/gcc-4.8.5 
module load intel-mkl/2020.2.254/intel-20.0.2
module load flac/1.3.2/gcc-9.2.0



source activate speech3.8



PYTHON="/gpfs/users/pellegrainv/.conda/envs/speech3.8/bin/python"

$PYTHON /gpfs/users/pellegrainv/RNN-Time-series-Anomaly-Detection/2_anomaly_detection.py --data ../../../workdir/pellegrainv/3Tanks/capteurs_labels_visible