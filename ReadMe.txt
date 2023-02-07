READ ME :D

For openMP run the following commands

1. Login into your Wave account
2. Copy artifacts in the same location from where you run the code
3. Run: $srun --nodes 1 --ntasks 1 --cpus-per-task 56 --mem=96G --pty /bin/bash
4. Run: $module load GCC
5: Run: make
Or 
5: Run: $g++ -fopenmp XRay_omp.c -o XRay_omp
6: Run: $./XRay_omp

Changes that can be made:
1. Name of the training set: dataset_filename (train.txt or train1.txt)


For CUDA run the following commands

1. Login into your Wave account
2. Copy artifacts in the same location from where you run the code
3. Run the following commands on the login node
4. Run: $module load CUDA
5: Run: make XRay_cuda
Or 
5: Run: $nvcc XRay_cuda.cu -o XRay_cuda
6: Run: $./XRay_cuda


