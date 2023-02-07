#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <string.h>


#define MAX_ITER 100
#define THRESHOLD 1e-6

// Global Variables
int num_pts_global;
int num_threads_global;
int num_iter_global;
int K_global;
int *data_pts_global;
float *iterative_centroids_global;
int *data_pts_cluster_global;
int **iterative_cluster_count_global;

// Define delta
double delta_global = THRESHOLD + 1;

void kmeans_thread(int *tid)
{
    int *id = (int *)tid;

    // Divide array into equal chunks and assign it to threads
    int data_length_per_thread = num_pts_global / num_threads_global;
    int start = (*id) * data_length_per_thread;
    int end = start + data_length_per_thread;
    if (end + data_length_per_thread > num_pts_global)
    {
        //Assign last undistributed points to this thread
        end = num_pts_global;
        data_length_per_thread = num_pts_global - start;
    }

    printf("Thread ID:%d, start:%d, end:%d\n", *id, start, end);

    int i = 0, j = 0;
    double min_dist, current_dist;

    // to assign cluster id to each point
    int *point_to_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));

    // to assign sum of all the cluster points in a cluster
    float *cluster_points_sum = (float *)malloc(K_global * 3 * sizeof(float));

    // to count the number of points in a cluster
    int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));

    int iter_counter = 0;
    while ((delta_global > THRESHOLD) && (iter_counter < MAX_ITER))
    {
        // Initialize cluster_points_sum to 0.0
        for (i = 0; i < K_global * 3; i++)
            cluster_points_sum[i] = 0.0;

        // Initializecluster count to 0
        for (i = 0; i < K_global; i++)
            points_inside_cluster_count[i] = 0;

        for (i = start; i < end; i++)
        {
            //assign cluster id of the nearest centriod
            min_dist = DBL_MAX;
            #pragma omp parallel for
            for (j = 0; j < K_global; j++)
            {
                current_dist = pow((double)(iterative_centroids_global[(iter_counter * K_global + j) * 3] - (float)data_pts_global[i * 3]), 2.0) +
                               pow((double)(iterative_centroids_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_pts_global[i * 3 + 1]), 2.0) +
                               pow((double)(iterative_centroids_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_pts_global[i * 3 + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    point_to_cluster_id[i - start] = j;
                }
            }

            //Localy update the count of number of points inside cluster
            points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;

            // Update local sum of cluster data points
            cluster_points_sum[point_to_cluster_id[i - start] * 3] += (float)data_pts_global[i * 3];
            cluster_points_sum[point_to_cluster_id[i - start] * 3 + 1] += (float)data_pts_global[i * 3 + 1];
            cluster_points_sum[point_to_cluster_id[i - start] * 3 + 2] += (float)data_pts_global[i * 3 + 2];
        }

/*
    Update the new centroid and new cluster count after each thread classification 
*/
#pragma omp critical
        {
            for (i = 0; i < K_global; i++)
            {
                if (points_inside_cluster_count[i] == 0)
                {
                    printf("Unlikely situation!\n");
                    continue;
                }
                iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3] = (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3] * iterative_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3]) / (float)(iterative_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] = (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] * iterative_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 1]) / (float)(iterative_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] = (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] * iterative_cluster_count_global[iter_counter][i] + cluster_points_sum[i * 3 + 2]) / (float)(iterative_cluster_count_global[iter_counter][i] + points_inside_cluster_count[i]);
                
                iterative_cluster_count_global[iter_counter][i] += points_inside_cluster_count[i];
            }
        }

/*
    Only after all the threads complete, a single thread should update the delta value between the old and new centroid
*/
#pragma omp barrier
        if (*id == 0)
        {
            double temp_delta = 0.0;
            for (i = 0; i < K_global; i++)
            {
                temp_delta += (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iterative_centroids_global[((iter_counter)*K_global + i) * 3]) * (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3] - iterative_centroids_global[((iter_counter)*K_global + i) * 3]) + (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iterative_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) * (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 1] - iterative_centroids_global[((iter_counter)*K_global + i) * 3 + 1]) + (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iterative_centroids_global[((iter_counter)*K_global + i) * 3 + 2]) * (iterative_centroids_global[((iter_counter + 1) * K_global + i) * 3 + 2] - iterative_centroids_global[((iter_counter)*K_global + i) * 3 + 2]);
            }
            delta_global = temp_delta;
            num_iter_global++;
        }

//  Only after all the threads complete, the local iterator should be incremented
#pragma omp barrier
        iter_counter++;
    }


// Assign datapoints along with its cluster id to the global cluster
    for (i = start; i < end; i++)
    {
        // Assign points to clusters
        data_pts_cluster_global[i * 4] = data_pts_global[i * 3];
        data_pts_cluster_global[i * 4 + 1] = data_pts_global[i * 3 + 1];
        data_pts_cluster_global[i * 4 + 2] = data_pts_global[i * 3 + 2];
        data_pts_cluster_global[i * 4 + 3] = point_to_cluster_id[i - start];
        assert(point_to_cluster_id[i - start] >= 0 && point_to_cluster_id[i - start] < K_global);
    }
}

void kmeans_omp(int num_threads,
                    int N,
                    int K,
                    int *data_pts,
                    int **data_points_cluster_id,
                    float **iterative_centroids,
                    int *num_of_iterations)
{


    //Global variables
    num_pts_global = N;
    num_threads_global = num_threads;
    num_iter_global = 0;
    K_global = K;
    data_pts_global = data_pts;

    *data_points_cluster_id = (int *)malloc(N * 4 * sizeof(int));   //Allocating space of 4 units each for N data points
    data_pts_cluster_global = *data_points_cluster_id;

    /*
       Allocate space for centroid in evry iteration
    */
    iterative_centroids_global = (float *)calloc((MAX_ITER + 1) * K * 3, sizeof(float));

    // Assigning random points to be initial centroids



    // for(int j = 0; j < sizeof(data_pts); j++){
    //     //printf("ELement %d is %d\n", j, data_pts[j]);
    // }

    int i = 0;

    iterative_centroids_global[0] = 11;
    iterative_centroids_global[1] = 11;
    iterative_centroids_global[2] = 11;
    iterative_centroids_global[3] = 109;
    iterative_centroids_global[4] = 109;
    iterative_centroids_global[5] = 109;
    //248
    // Print initial centroids
    for (i = 0; i < K; i++)
    {
        printf("initial centroid #%d: %f,%f,%f\n", i + 1, iterative_centroids_global[i * 3], iterative_centroids_global[i * 3 + 1], iterative_centroids_global[i * 3 + 2]);
    }

    /*
        Allocate space to store the cluster count after each iteration
     */
    iterative_cluster_count_global = (int **)malloc(MAX_ITER * sizeof(int *));
    for (i = 0; i < MAX_ITER; i++)
    {
        iterative_cluster_count_global[i] = (int *)calloc(K, sizeof(int));
    }

    // Set number of threads
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("Thread: %d created!\n", ID);
        kmeans_thread(&ID);
    }

    // Store num_of_iterations
    *num_of_iterations = num_iter_global;

    //Store the centroids global for each iteration into local centroids
    int iterative_centroids_size = (*num_of_iterations + 1) * K * 3;
    printf("Number of iterations :%d\n", *num_of_iterations);
    *iterative_centroids = (float *)calloc(iterative_centroids_size, sizeof(float));
    for (i = 0; i < iterative_centroids_size; i++)
    {
        (*iterative_centroids)[i] = iterative_centroids_global[i];
    }

    // Print final centroids after last iteration
    for (i = 0; i < K; i++)
    {
        printf("centroid #%d: %f,%f,%f\n", i + 1, (*iterative_centroids)[((*num_of_iterations) * K + i) * 3], (*iterative_centroids)[((*num_of_iterations) * K + i) * 3 + 1], (*iterative_centroids)[((*num_of_iterations) * K + i) * 3 + 2]);
    }

}

void dataset_in(char *dataset_filename, int *N, int **data_pts)
{
	FILE *fin = fopen(dataset_filename, "r");
	fscanf(fin, "%d", N);
	*data_pts = (int *)malloc(sizeof(int) * ((*N) * 3));
    int i = 0;
	for (i = 0; i < (*N) * 3; i++)
	{
		fscanf(fin, "%d", (*data_pts + i));
	}
    printf("N is %d\n", *N);
	fclose(fin);
}


void clusters_out(const char *cluster_filename, int N, int *cluster_pts)
{
	FILE *fout = fopen(cluster_filename, "w");
    int i = 0;
	for (i = 0; i < N; i++)
	{
		fprintf(fout, "%d %d %d %d\n",
				*(cluster_pts + (i * 4)), *(cluster_pts + (i * 4) + 1),
				*(cluster_pts + (i * 4) + 2), *(cluster_pts + (i * 4) + 3));
	}
	fclose(fout);
}

void centroids_out(const char *centroid_filename, int K, int num_of_iterations, float *iterative_centroids)
{
	FILE *fout = fopen(centroid_filename, "w");
    int i = 0;
	for (i = 0; i < num_of_iterations + 1; i++)
	{
        int j = 0;
		for (j = 0; j < K; j++)
		{
			fprintf(fout, "%f %f %f, ",
					*(iterative_centroids + (i * K + j) * 3),		 //R value
					*(iterative_centroids + (i * K + j) * 3 + 1),  //G value
					*(iterative_centroids + (i * K + j) * 3 + 2)); //B value
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

void train(char *dataset_filename){

	int N;					//total number of data points
	int K;					//K cluster value
	int num_threads;	
	int* data_pts;		//storing data points
	int* cluster_pts;	//data points after clustering
	float* iterative_centroids;			//previous centroids
	int num_of_iterations;  


    K=2;
    printf("Enter No. of threads to be used: ");
    scanf("%d",&num_threads);


	/*
  			This function will essentially read and store data points into the array.
	*/

	dataset_in (dataset_filename, &N, &data_pts);

    /*
    		The centroids will be stored for each iteration in a 1D array
        [iter_1_cluster_1_x, iter_1_cluster_1_y, iter_1_cluster_1_z, iter_1_cluster_2_x, iter_1_cluster_2_y, iter_1_cluster_2_z, iter_2_cluster_1_x, ...]
    */


    kmeans_omp(num_threads, N, K, data_pts, &cluster_pts, &iterative_centroids, &num_of_iterations);

    char cluster_filename[105] = "cluster_output_threads";
    strcat(cluster_filename,"_dataset");
    strcat(cluster_filename,".txt");

    char centroid_filename[105] = "centroid_output_threads";
    strcat(centroid_filename,"_dataset");
    strcat(centroid_filename,".txt");

	/*
  			This function saves the computed clusters in a file
	*/
	clusters_out (cluster_filename, N, cluster_pts);
	centroids_out (centroid_filename, K, num_of_iterations, iterative_centroids);
    
	char time_file_omp[100] = "compute_time_openmp_threads";
    strcat(time_file_omp,"_dataset");
    strcat(time_file_omp,".txt");
    
    
	printf("Centroids of clusters stored in:  '%s' saved\n", centroid_filename);
    printf("Cluster points stored in:  '%s' saved\n", cluster_filename);
}

void test(char *dataset_filename){

	int N;					//total number of data points
	int K;					//K cluster value
	int num_threads;	
	int* data_pts;		//storing data points
	int* cluster_pts;	//data points after clustering
	float* iterative_centroids;			//previous centroids
	int num_of_iterations;  
	
   

    printf("\nInside test! \n");
    K=2;
    printf("Enter No. of threads to be used: ");
    scanf("%d",&num_threads);

    

    dataset_in (dataset_filename, &N, &data_pts);
    char line[1024]={0,}; // Initialize memory! You have to do this (as for your question)
    FILE *sync = fopen("centroid_output_threads_dataset.txt", "r");
    if( sync ) {
      while( fgets(line, 1024, sync) !=NULL ) {
      // Just search for the latest line, do nothing in the loop
      } 
      fclose(sync);
    }
   float *trained_global;
   trained_global = (float *)calloc(6, sizeof(float));
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
        // for( i = 0; i < 6; i++){
        //     printf("%f\n", trained_global[i]);
        // }
    int num_threads_global = num_threads;
    omp_set_num_threads(num_threads);
    double start_time, end_time;
	double diff;
    FILE *ptr;
    ptr = fopen("/WAVE/users/unix/rbaskar/Project/log.log","a+");

    start_time =omp_get_wtime();
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        int *id = (int *) &ID;
        num_pts_global = N;
        K_global = K;


    int data_length_per_thread = num_pts_global / num_threads_global;
    int start = (*id) * data_length_per_thread;
    int end = start + data_length_per_thread;
    if (end + data_length_per_thread > num_pts_global)
    {
        //for the last chunk of data
        end = num_pts_global;
        data_length_per_thread = num_pts_global - start;
    }

    int i = 0, j = 0;
    double min_dist, current_dist;

    // cluster classification storage
    int *point_to_cluster_id = (int *)malloc(data_length_per_thread * sizeof(int));
    int *points_inside_cluster_count = (int *)malloc(K_global * sizeof(int));
    // printf("assigned space for cluster id and cluster count %d ID\n", *id);
    
    for (i = 0; i < K_global; i++)
        points_inside_cluster_count[i] = 0;
    data_pts_global = data_pts;
    // printf("After assigning default cluster id %d ID, start: %d, end: %d\n",*id, start, end);
    for (i = start; i < end; i+=3)
        {
            min_dist = DBL_MAX;
            int iter_counter =0;
            #pragma omp parallel for
            for (j = 0; j < K_global; j++)
            {
                current_dist = pow((double)(trained_global[(iter_counter * K_global + j) * 3] - (float)data_pts_global[i]), 2.0) +
                               pow((double)(trained_global[(iter_counter * K_global + j) * 3 + 1] - (float)data_pts_global[i + 1]), 2.0) +
                               pow((double)(trained_global[(iter_counter * K_global + j) * 3 + 2] - (float)data_pts_global[i + 2]), 2.0);
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    point_to_cluster_id[i - start] = j;
                }
                
            }
            points_inside_cluster_count[point_to_cluster_id[i - start]] += 1;
        }
    #pragma omp barrier
    if(*id == 0)
        if(points_inside_cluster_count[0]>points_inside_cluster_count[1]){
            printf("Patient belongs to class 1\n");
            printf("Patient's X-ray is normal\n");
        }     
        else{
            printf("Patient belongs to class 2\n");
            printf("Patient's X-ray shows Pneumonia\n");}
    }
    end_time =omp_get_wtime();
    diff=(end_time-start_time);
    printf("Time Taken is %f seconds\n", diff);
    fprintf(ptr, "Time Taken is %f seconds\n", diff);
    fclose(ptr);
}

int main()
{    
    int option;
    int num_threads;
    int N;

    printf("----------------\n");
    printf("1. TRAIN\n2. TEST\n");
    
    printf("ENTER YOUR OPTION: ");
    scanf("%d", &option);
    

    switch (option)
    {
    case 1:
        train("train1.txt");
        break;
    
    case 2:
        test("testN1.txt");
        test("testP1.txt");
        test("testN2.txt");
        break;
    
    default:
        test("testN1.txt");
        test("testP1.txt");
        test("testN2.txt");
    }	
}