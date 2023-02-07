#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>


#define MAX_ITER 100
#define THRESHOLD 1e-6

__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

__device__ void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
  }

// Global Variables used across different functions
__device__ __managed__ int number_of_points_global;
__device__ __managed__ int number_of_threads_global;
__device__ __managed__ int number_of_iterations_global;
__device__ __managed__ int K_global;
__device__ __managed__ int *data_points_global;
__device__ __managed__ float *iter_centroids_global;
__device__ __managed__ int *data_point_cluster_global;
__device__ __managed__ int **iter_cluster_count_global;
__device__ __managed__ float *trained_global;

// Defined global delta
__managed__ double delta_global = THRESHOLD + 1;

__global__ void kmeans_cuda_thread(){
    int id = threadIdx.x;
    printf("This is thread %d \n", id);

    if(id == 0){
        printf("Iter centroid global is %f \n", iter_centroids_global[0]);
    }
    
    int data_length_per_thread = number_of_points_global / number_of_threads_global;
    int start = (id) * data_length_per_thread;
    int end = start + data_length_per_thread;
    if (end + data_length_per_thread > number_of_points_global)
    {
        end = number_of_points_global;
        data_length_per_thread = number_of_points_global - start;
    }

    printf("Thread ID:%d, start:%d, end:%d\n", id, start, end);

    int i = 0, j = 0;
    double min_dist, current_dist;

    int *point_to_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));

    float *cluster_points_sum = (float *)malloc(K_global * 3 * sizeof(float));

    int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));

    int iter_counter = 0;
    while ((delta_global > THRESHOLD) && (iter_counter < MAX_ITER))
    {
        for (i = 0; i < K_global * 3; i++){
            cluster_points_sum[i] = 0.0;
            //printf("Cluster points sum %d is %f \n", i, cluster_points_sum[i]);
        }
        
        for (i = 0; i < K_global; i++){
            points_inside_cluster_count[i] = 0;
            //printf("Points inside cluster count %d is %f \n", i, points_inside_cluster_count[i]);
        }

        for (i = start; i < end; i++)
        {

            min_dist = DBL_MAX;
            for (j = 0; j < K_global; j++)
            {   


                current_dist = pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3] - (float)data_points_global[i * 3]), 2.0) +
                               pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_points_global[i * 3 + 1]), 2.0) +
                               pow((double)(iter_centroids_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_points_global[i * 3 + 2]), 2.0);
                
                printf("Current distance is %f \n", current_dist);

                if (current_dist < min_dist)
                {   
                    
                    min_dist = current_dist;
                    point_to_cluster_id[i - start] = j;
                    printf("\n DONE!!! \n");
                    
                }

                
            }

            points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;

            cluster_points_sum[point_to_cluster_id[i - start] * 3] += (float)data_points_global[i * 3];
            cluster_points_sum[point_to_cluster_id[i - start] * 3 + 1] += (float)data_points_global[i * 3 + 1];
            cluster_points_sum[point_to_cluster_id[i - start] * 3 + 2] += (float)data_points_global[i * 3 + 2];
        }

    acquire_semaphore(&sem);
    __syncthreads();
        {
            for (i = 0; i < K_global; i++)
            {
                if (points_inside_cluster_count[i] == 0)
                {
                    printf("Unlikely situation!\n");
                    continue;
                }
                iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 1]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] = (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] * iter_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 2]) / (float)(iter_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                
                iter_cluster_count_global[iter_counter][i] += points_inside_cluster_count[i];
            }
        }

    __threadfence(); 
    __syncthreads();
    release_semaphore(&sem);

    __syncthreads();
        if (id == 0)
        {
            double temp_delta = 0.0;
            for (i = 0; i < K_global; i++)
            {
                temp_delta += (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iter_centroids_global[((iter_counter)*K_global + i) * 3]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) + (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]) * (iter_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iter_centroids_global[((iter_counter)*K_global + i) * 3 + 2]);
            }
            delta_global = temp_delta;
            number_of_iterations_global++;
        }

    __syncthreads();
        iter_counter++;
    }
    for (i = start; i < end; i++)
    {
        
        data_point_cluster_global[i * 4] = data_points_global[i * 3];
        data_point_cluster_global[i * 4 + 1] = data_points_global[i * 3 + 1];
        data_point_cluster_global[i * 4 + 2] = data_points_global[i * 3 + 2];
        data_point_cluster_global[i * 4 + 3] = point_to_cluster_id[i - start];
        assert(point_to_cluster_id[i - start] >= 0 && point_to_cluster_id[i - start] < K_global);
    }


}

void kmeans_cuda(int num_threads,
                    int N,
                    int K,
                    int *data_points,
                    int **data_points_cluster_id,
                    float **iter_centroids,
                    int *number_of_iterations)
{


    // Initialize global variables
    number_of_points_global = N;
    number_of_threads_global = num_threads;
    number_of_iterations_global = 0;
    K_global = K;
    data_points_global = data_points;

    

    cudaMallocManaged(&data_points_cluster_id, N * 4 * sizeof(int));   //Allocating space of 4 units each for N data points
    data_point_cluster_global = *data_points_cluster_id;

  //  *data_points_cluster_id = (int *)malloc(N * 4 * sizeof(int));   //Allocating space of 4 units each for N data points
 
    cudaMallocManaged(&iter_centroids_global, (MAX_ITER + 1) * K * 3 * sizeof(float));
    cudaMemset(&iter_centroids_global, 0, (MAX_ITER + 1) * K * 3 * sizeof(float));
    //iter_centroids_global = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    int i = 0;

    iter_centroids_global[0] = 11;
    iter_centroids_global[1] = 11;
    iter_centroids_global[2] = 11;
    iter_centroids_global[3] = 109;
    iter_centroids_global[4] = 109;
    iter_centroids_global[5] = 109;

    // Print initial centroids
    for (i = 0; i < K; i++)
    {
        printf("initial centroid #%d: %f,%f,%f\n", i + 1, iter_centroids_global[i * 3], iter_centroids_global[i * 3 + 1], iter_centroids_global[i * 3 + 2]);
    }

    cudaMallocManaged(&*iter_cluster_count_global, MAX_ITER * sizeof(int *));

    for (i = 0; i < MAX_ITER; i++)
    {   
        cudaMallocManaged(&iter_cluster_count_global[i], K * sizeof(int));
        cudaMemset(iter_cluster_count_global[i], 0, K * sizeof(int));
        //iter_cluster_count_global[i] = (int *)calloc(K, sizeof(int));
    }

    // int ID = omp_get_thread_num();
    // printf("Thread: %d created!\n", ID);
    // kmeans_openmp_thread(&ID);

    kmeans_cuda_thread<<<1,number_of_threads_global>>>();

    cudaDeviceSynchronize();

    *number_of_iterations = number_of_iterations_global;

    int iter_centroids_size = (*number_of_iterations + 1) * K * 3;
    printf("Number of iterations :%d\n", *number_of_iterations);

    //*iter_centroids = (float *)calloc(iter_centroids_size, sizeof(float));
    
    cudaMallocManaged(&iter_centroids, iter_centroids_size * sizeof(float));
    cudaMemset(&iter_centroids, 0, iter_centroids_size * sizeof(float));

    for(int i = 0; i < iter_centroids_size; i++){
        printf("%d : %f\n", i, iter_centroids[i]);
    }

    for (i = 0; i < iter_centroids_size; i++)
    {
        printf("%d : %f", i, iter_centroids_global[i]);
        (*iter_centroids)[i] = iter_centroids_global[i];
        printf("%d : %f", i, iter_centroids[i]);
    }

    for (i = 0; i < K; i++)
    {
        printf("centroid #%d: %f,%f,%f\n", i + 1, (*iter_centroids)[((*number_of_iterations) * K + i) * 3], (*iter_centroids)[((*number_of_iterations) * K + i) * 3 + 1], (*iter_centroids)[((*number_of_iterations) * K + i) * 3 + 2]);
    }

}

void dataset_in(char *dataset_filename, int *N, int **data_points)
{
	FILE *fin = fopen(dataset_filename, "r");
	fscanf(fin, "%d", N);
	cudaMallocManaged(&*data_points, sizeof(int) * ((*N) * 3));
    //*data_points = (int *)malloc(sizeof(int) * ((*N) * 3));
    int i = 0;
	for (i = 0; i < (*N) * 3; i++)
	{
		fscanf(fin, "%d", (*data_points + i));
	}
    printf("After reading\n");
    printf("N is %d\n", *N);
    printf("Before closing\n");
	fclose(fin);
    printf("Closed\n");
}

void clusters_out(const char *cluster_filename, int N, int *cluster_points)
{
	FILE *fout = fopen(cluster_filename, "w");
    int i = 0;
	for (i = 0; i < N; i++)
	{
		fprintf(fout, "%d %d %d %d\n",
				*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
				*(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
	}
	fclose(fout);
}

void centroids_out(const char *centroid_filename, int K, int number_of_iterations, float *iter_centroids)
{
	FILE *fout = fopen(centroid_filename, "w");
    int i = 0;
	for (i = 0; i < number_of_iterations + 1; i++)
	{
        int j = 0;
		for (j = 0; j < K; j++)
		{
			fprintf(fout, "%f %f %f, ",
					*(iter_centroids + (i * K + j) * 3),		 
					*(iter_centroids + (i * K + j) * 3 + 1),  
					*(iter_centroids + (i * K + j) * 3 + 2)); 
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}



void train(){

    printf("Inside TRAIN \n");

	int N;					
	int K;					
	int num_threads;		
	int* data_points;		
	int* cluster_points;	
	float* iter_centroids;			
	int number_of_iterations;    

    char *dataset_filename = "train.txt";
    //CHANGE to 2 clusters and max threaeds
    printf("Enter No. of Clusters: ");
    scanf("%d", &K);
    printf("Enter No. of threads to be used: ");
    scanf("%d",&num_threads);

    int x = 10;

	//double start_time, end_time;
	//double computation_time;

	dataset_in (dataset_filename, &N, &data_points);

	//start_time = omp_get_wtime();
    kmeans_cuda(num_threads, N, K, data_points, &cluster_points, &iter_centroids, &number_of_iterations);
	//end_time = omp_get_wtime();

    char num_threads_char[3];
    snprintf(num_threads_char,10,"%d", num_threads);

    char file_index_char[2];
    snprintf(file_index_char,10,"%d", x);

    char cluster_filename[105] = "cluster_output_threads";
    strcat(cluster_filename,num_threads_char);
    strcat(cluster_filename,"_dataset");
    strcat(cluster_filename,file_index_char);
    strcat(cluster_filename,".txt");

    char centroid_filename[105] = "centroid_output_threads";
    strcat(centroid_filename,num_threads_char);
    strcat(centroid_filename,"_dataset");
    strcat(centroid_filename,file_index_char);
    strcat(centroid_filename,".txt");

	clusters_out (cluster_filename, N, cluster_points);

	centroids_out (centroid_filename, K, number_of_iterations, iter_centroids);

    
	char time_file_omp[100] = "compute_time_openmp_threads";
    strcat(time_file_omp,num_threads_char);
    strcat(time_file_omp,"_dataset");
    strcat(time_file_omp,file_index_char);
    strcat(time_file_omp,".txt");

	FILE *fout = fopen(time_file_omp, "a");
	//fprintf(fout, "%f\n", computation_time);
	fclose(fout);
    
	printf("Cluster Centroid point output file '%s' saved\n", centroid_filename);
    printf("Clustered points output file '%s' saved\n", cluster_filename);
    printf("Computation time output file '%s' saved\n", time_file_omp);


}


__global__ void test_helper(){

            //printf("Inside helper function! \n");
            int id = threadIdx.x;
          //  printf("This is thread %d \n", id);

            
            int data_length_per_thread = number_of_points_global / number_of_threads_global;
            int start = (id) * data_length_per_thread;
            int end = start + data_length_per_thread;
            if (end + data_length_per_thread > number_of_points_global)
            {
                end = number_of_points_global;
                data_length_per_thread = number_of_points_global - start;
            }

            // printf("Thread ID:%d, start:%d, end:%d\n", *id, start, end);
            int i = 0, j = 0;
            double min_dist, current_dist;

            int *point_to_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));
            int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));
            // printf("assigned space for cluster id and cluster count %d ID\n", *id);
            for (i = 0; i < K_global; i++)
                points_inside_cluster_count[i] = 0;
            
            // printf("After assigning default cluster id %d ID, start: %d, end: %d\n",*id, start, end);
            for (i = start; i < end; i+=3)
                {
                    min_dist = DBL_MAX;
                    // printf("HERE! ID: %d\n", *id);
                    
                    // printf("HERE!! ID: %d\n", *id);
                    int iter_counter =0;
                    for (j = 0; j < K_global; j++)
                    {
                        // printf("ID: %d\n",*id);
                        // printf("ID: %d, data_points_global: %d\n",*id, i);
                        current_dist = pow((double)(trained_global[(iter_counter * K_global + j) * 3] - (float)data_points_global[i]), 2.0) +
                                    pow((double)(trained_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_points_global[i + 1]), 2.0) +
                                    pow((double)(trained_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_points_global[i + 2]), 2.0);
                        // printf("End ID: %d\n",*id);
                        if (current_dist < min_dist)
                        {
                            min_dist = current_dist;
                            point_to_cluster_id[i - start] = j;
                        }
                        
                    }
                    
                    points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;
                }
            // printf("After finding the distancess id %d ID\n",*id);
            __syncthreads();
            if(id == 0)
                if(points_inside_cluster_count[0]>points_inside_cluster_count[1]){
                    printf("Patient belongs to class 1\n");
                    printf("Patient's X-ray is normal\n");
                }     
                else{
                    printf("Patient belongs to class 2\n");
                    printf("Patient's X-ray shows Pneumonia\n");}
}


void test(){

   // printf("Inside TEST! \n");

	int N;					
	int K;					
	int num_threads;		
	int* data_points;		
	int* cluster_points;	
	float* iter_centroids;		
	int number_of_iterations;    
   

    
    printf("\nInside test! \n");
    char *dataset_filename = "testP1.txt";
    //CHANGE to 2 clusters and max threaeds
    // printf("Enter No. of Clusters: ");
    // scanf("%d", &K);
    K=2;
    printf("Enter No. of threads to be used: ");
    scanf("%d",&num_threads);

    double start_time, end_time;
	double computation_time;

    dataset_in (dataset_filename, &N, &data_points);
    printf("After dataset in\n");
    char line[1024]={0,}; // Initialize memory! You have to do this (as for your question)
    FILE *sync = fopen("centroid_output_threads_dataset10.txt", "r");
    if( sync ) {
      while( fgets(line, 1024, sync) !=NULL ) {
      // Just search for the latest line, do nothing in the loop
      } 
      //printf("Last line %s\n", line); //<this is just a log... you can remove it
      fclose(sync);
    }
   

   cudaMallocManaged(&trained_global, 6 * sizeof(float));
   cudaMemset(&trained_global, 0, 6 * sizeof(float));
   
   char * token = NULL;
		// split the input record and add it to the data packet
		token = strtok(line," ");
        trained_global[0] = atof(token);
        
        token = strtok(NULL," ");
        trained_global[1] = atof(token);
        
        token = strtok(NULL," ");
        trained_global[2] = atof(token);

        token = strtok(NULL," ");
        trained_global[3] = atof(token);

        token = strtok(NULL," ");
        trained_global[4] = atof(token);

        token = strtok(NULL," ");
        trained_global[5] = atof(token);
        int i;
        for( i = 0; i < 6; i++){
            printf("%f\n", trained_global[i]);
        }
    int number_of_threads_global = num_threads;
    number_of_points_global = N;
    K_global = K;
    data_points_global = data_points;

    //printf("Now calling helper function\n");

    cudaEvent_t start, stop;
float gpu_time = 0.0f;
cudaEventCreate(&start) ;
 cudaEventCreate(&stop) ;
cudaEventRecord(start, 0);


    test_helper<<<1,number_of_threads_global>>>();


    cudaThreadSynchronize();
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
 cudaEventElapsedTime(&gpu_time, start, stop) ;
printf("Time spent: %.5f\n", gpu_time);
cudaEventDestroy(start);
cudaEventDestroy(stop);

    cudaDeviceSynchronize();
}


int main()
{    
    int option;


    printf("----------------\n");
    printf("1. TRAIN\n2. TEST\n");
    
    printf("ENTER YOUR OPTION: ");
    scanf("%d", &option);

    switch (option)
    {
    case 1:
        train();
        break;
    
    case 2:
        test();
        break;
    
    default:
        test();
        
    }	
}