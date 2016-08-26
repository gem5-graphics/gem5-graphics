#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
//#define NUM_THREAD 4
#define OPEN

#ifdef GEM5_WORK
#include <stdint.h>
extern "C" {
void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
void m5_work_begin(uint64_t workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

#ifdef TIMING
#include <sys/time.h>
double gettime() {
    struct timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec+t.tv_usec*1e-6;
}
#endif

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
	char *input_f;
	int	 num_omp_threads;
	
	if(argc!=3){
	Usage(argc, argv);
	exit(0);
	}

	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);


	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	

#ifdef OPEN
    omp_set_num_threads(num_omp_threads);
#endif

#ifdef GEM5_WORK
	m5_work_begin(0, 0);
	m5_dumpreset_stats(0, 0);
#endif

	printf("Start traversing the tree\n");
	
#ifdef TIMING
    double start_time = gettime();
#endif

    int k=0;

	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

#ifdef OPEN
		#pragma omp parallel for 
#endif 
		for(int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
			h_graph_mask[tid]=false;
			for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++)
				{
				int id = h_graph_edges[i];
				if(!h_graph_visited[id])
					{
					h_cost[id]=h_cost[tid]+1;
					h_updating_graph_mask[id]=true;
					}
				}
			}
		}

#ifdef OPEN
        #pragma omp parallel for shared(stop)
#endif
  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
			h_graph_mask[tid]=true;
			h_graph_visited[tid]=true;
			stop=true;
			h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);

#ifdef TIMING
    double end_time = gettime();
    printf("ROI Runtime: %f\n", end_time - start_time);
#endif

    printf("Kernel Executed %d times\n",k);

#ifdef GEM5_WORK
	m5_dumpreset_stats(0, 0);
	m5_work_end(0, 0);
#endif

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
//	FILE *fpo = stdout;
	for(int i=0;i<no_of_nodes&&i<1000;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);

}

