srun -p moss --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=32 --mem-per-cpu=2G --output=moss004.log --job-name=tritons01 python s_train_scratch.py
srun -p a800 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=32 --mem-per-cpu=2G --output=moss005.log --job-name=tritons01 python s_train_o.py