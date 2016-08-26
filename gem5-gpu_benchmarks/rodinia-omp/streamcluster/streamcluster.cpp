/***********************************************
    streamcluster_omp.cpp
    : parallelized code of streamcluster using OpenMP

    - original code from PARSEC Benchmark Suite
    - parallelization with OpenMP API has been applied by

    Sang-Ha (a.k.a Shawn) Lee - sl4ge@virginia.edu
    University of Virginia
    Department of Electrical and Computer Engineering
    Department of Computer Science

***********************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>
#include <omp.h>

#ifdef GEM5_WORK
#include <stdint.h>
extern "C" {
  void m5_dumpreset_stats(uint64_t ns_delay, uint64_t ns_period);
  void m5_work_begin(uint64_t workid, uint64_t threadid);
  void m5_work_end(uint64_t workid, uint64_t threadid);
}
#endif

using namespace std;

#define MAXNAMESIZE 1024 // max filename length
#define SEED 1
/* increase this to reduce probability of random error */
/* increasing it also ups running time of "speedy" part of the code */
/* SP = 1 seems to be fine */
#define SP 1 // number of repetitions of speedy must be >=1

/* higher ITER --> more likely to get correct # of centers */
/* higher ITER also scales the running time almost linearly */
#define ITER 3 // iterate ITER* k log k times; ITER >= 1

#define CACHE_LINE 512 // cache line in byte

/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
    float weight;
    float *coord;
    long assign;  /* number of point where this one is assigned */
    float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
    long num; /* number of points; may not be N if this is a sample */
    int dim;  /* dimensionality */
    Point *p; /* the array itself */
} Points;

static bool *switch_membership; //whether to switch membership in pgain
static bool* is_center; //whether a point is a center
static int* center_table; //index table of centers
float* block;

static int ompthreads;

// instrumentation code
#ifdef TIMING
double time_local_search;
int iters_local_search;
double time_speedy;
int iters_speedy;
double time_select_feasible;
int iters_select_feasible;
double time_shuffle;
int iters_shuffle;
double time_pgain;
int iters_pgain;
double time_pgain_dist;
double time_pgain_init;

double
gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + t.tv_usec * 1e-6;
}
#endif

// tells whether two points of D dimensions are identical
int
isIdentical(float *i, float *j, int D)
{
    int a = 0;
    int equal = 1;

    while (equal && a < D) {
        if (i[a] != j[a]) equal = 0;
        else a++;
    }

    if (equal) return 1;
    else return 0;
}

/* shuffle points into random order */
void
shuffle(Points *points)
{
#ifdef TIMING
    double t1 = gettime();
    iters_shuffle++;
#endif
    long i, j;
    Point temp;
    for (i = 0; i < points->num - 1; i++) {
        j = (lrand48() % (points->num - i)) + i;
        temp = points->p[i];
        points->p[i] = points->p[j];
        points->p[j] = temp;
    }
#ifdef TIMING
    double t2 = gettime();
    time_shuffle += t2-t1;
#endif
}

/* shuffle an array of integers */
void
intshuffle(int *intarray, int length)
{
#ifdef TIMING
    double t1 = gettime();
    iters_shuffle++;
#endif
    long i, j;
    int temp;
    for (i = 0; i < length; i++) {
        j = (lrand48() % (length - i))+i;
        temp = intarray[i];
        intarray[i] = intarray[j];
        intarray[j] = temp;
    }
#ifdef TIMING
    double t2 = gettime();
    time_shuffle += t2-t1;
#endif
}

/* compute Euclidean distance squared between two points */
float
dist(Point p1, Point p2, int dim)
{
    float result = 0.0;
    for (int i = 0; i < dim; i++)
        result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);
    return result;
}

/* run speedy on the points, return total cost of solution */
float
pspeedy(Points *points, float z, long *kcenter)
{
#ifdef TIMING
    double t1 = gettime();
    iters_speedy++;
#endif

    float totalcost;

    /* create center at first point, send it to itself */
    for (int k = 0; k < points->num; k++) {
        float distance = dist(points->p[k], points->p[0], points->dim);
        points->p[k].cost = distance * points->p[k].weight;
        points->p[k].assign = 0;
    }

    *kcenter = 1;
    
    for (int i = 1; i < points->num; i++) {
        bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z);
        if (to_open) {
            (*kcenter)++;
            for (int k = 0; k < points->num; k++) {
                float distance = dist(points->p[i], points->p[k], points->dim);
                if (distance*points->p[k].weight < points->p[k].cost) {
                    points->p[k].cost = distance * points->p[k].weight;
                    points->p[k].assign = i;
                }
            }
        }
    }

    totalcost = z * (*kcenter);
    for (int k = 0; k < points->num; k++) {
        totalcost += points->p[k].cost;
    }

#ifdef TIMING
    double t2 = gettime();
    time_speedy += t2 - t1;
#endif

    return totalcost;
}


/* For a given point x, find the cost of the following operation:
 * -- open a facility at x if there isn't already one there,
 * -- for points y such that the assignment distance of y exceeds dist(y, x),
 *    make y a member of x,
 * -- for facilities y such that reassigning y and all its members to x 
 *    would save cost, realize this closing and reassignment.
 * 
 * If the cost of this operation is negative (i.e., if this entire operation
 * saves cost), perform this operation and return the amount of cost saved;
 * otherwise, do nothing.
 */

/* numcenters will be updated to reflect the new number of centers */
/* z is the facility cost, x is the number of this point in the array 
   points */

float
pgain(long x, Points *points, float z, long int *numcenters)
{
#ifdef TIMING
    double t0 = gettime();
    iters_pgain++;
#endif

    int number_of_centers_to_close = 0;

    float gl_cost_of_opening_x;
    int gl_number_of_centers_to_close;

    //each thread takes a block of working_mem.
    int stride = *numcenters + 2;
    //make stride a multiple of CACHE_LINE
    int cl = CACHE_LINE/sizeof(float);
    if (stride % cl != 0) {
        stride = cl * (stride / cl + 1);
    }
    int K = stride - 2;

    float cost_of_opening_x = 0;

    float *work_mem = (float*) malloc(stride * 2 * sizeof(float));
    gl_cost_of_opening_x = 0;
    gl_number_of_centers_to_close = 0;

    /*
    * For each center, we have a *lower* field that indicates how much we will
    * save by closing the center. Each thread has its own copy of the *lower*
    * fields as an array. We first build a table to index the positions of the
    * *lower* fields.
    */

    // Assign unique IDs to current centers in center_table. This uniqueness
    // is important for ensuring race-free execution in OMP for loop below
    int count = 0;
    for (int i = 0; i < points->num; i++) {
        if (is_center[i]) {
            center_table[i] = count++;
        }
    }

    //now we finish building the table. clear the working memory.
    memset(switch_membership, 0, points->num * sizeof(bool));
    memset(work_mem, 0, 2 * stride * sizeof(float));

#ifdef TIMING
    double t1 = gettime();
    time_pgain_init += t1 - t0;
#endif
    //my *lower* fields
    float* lower = &work_mem[0];
    //global *lower* fields
    float* gl_lower = &work_mem[stride];

    float* my_lower;

    // OpenMP parallelization
    #pragma omp parallel shared(lower) private(my_lower)
    {
        my_lower = (float*) malloc(stride * sizeof(float));
        memset(my_lower, 0, stride * sizeof(float));

        #pragma omp for reduction(+: cost_of_opening_x)
        for (int i = 0; i < points->num; i++) {
            float x_cost = dist(points->p[i], points->p[x], points->dim) *
                           points->p[i].weight;
            float current_cost = points->p[i].cost;

            if (x_cost < current_cost) {
                // point i would save cost just by switching to x
                // (note that i cannot be a median,
                // or else dist(p[i], p[x]) would be 0)
                switch_membership[i] = true;
                cost_of_opening_x += x_cost - current_cost;
            } else {
                // cost of assigning i to x is at least current assignment cost of i
                // consider the savings that i's **current** median would realize
                // if we reassigned that median and all its members to x;
                // note we've already accounted for the fact that the median
                // would save z by closing; now we have to subtract from the savings
                // the extra cost of reassigning that median and its members
                int assign = points->p[i].assign;
                my_lower[center_table[assign]] += current_cost - x_cost;
            }
        }

        // Critical section seems to perform better, probably due to simpler
        // memory ordering than OMP atomic, which probably uses more locking
        #pragma omp critical
        for (int i = 0; i < stride; i++) {
            lower[i] += my_lower[i];
        }

        free(my_lower);
    }

#ifdef TIMING
    double t2 = gettime();
    time_pgain_dist += t2 - t1;
#endif

    // at this time, we can calculate the cost of opening a center
    // at x; if it is negative, we'll go through with opening it
    // NOTE: center_table contains unique center IDs to ensure race-free
    // updates to gl_lower here.
    #pragma omp parallel for reduction(+: cost_of_opening_x, number_of_centers_to_close)
    for (int i = 0; i < points->num; i++) {
        if (is_center[i]) {
            float low = z;
            //aggregate from all threads
            low += work_mem[center_table[i]];
            gl_lower[center_table[i]] = low;
            if (low > 0) {
                // i is a median, and
                // if we were to open x (which we still may not) we'd close i
                // note, we'll ignore the following quantity unless we do open x
                ++number_of_centers_to_close;
                cost_of_opening_x -= low;
            }
        }
    }

    //use the rest of working memory to store the following
    work_mem[K] = number_of_centers_to_close;
    work_mem[K+1] = cost_of_opening_x;

    gl_cost_of_opening_x = z;
    //aggregate
    gl_number_of_centers_to_close += (int)work_mem[K];
    gl_cost_of_opening_x += work_mem[K+1];

    // Now, check whether opening x would save cost; if so, do it, and
    // otherwise do nothing

    if (gl_cost_of_opening_x < 0) {
        //  we'd save money by opening x; we'll do it
        #pragma omp parallel for
        for (int i = 0; i < points->num; i++) {
            bool close_center = gl_lower[center_table[points->p[i].assign]] > 0;
            if (switch_membership[i] || close_center) {
                // Either i's median (which may be i itself) is closing,
                // or i is closer to x than to its current median
                points->p[i].cost = points->p[i].weight *
                                dist(points->p[i], points->p[x], points->dim);
                points->p[i].assign = x;
            }

            if (is_center[i] && gl_lower[center_table[i]] > 0) {
                is_center[i] = false;
            }
        }

        if (x >= 0 && x < points->num) {
            is_center[x] = true;
        }

        *numcenters = *numcenters + 1 - gl_number_of_centers_to_close;
    } else {
        gl_cost_of_opening_x = 0;  // the value we'll return
    }

    free(work_mem);

#ifdef TIMING
    double t3 = gettime();
    time_pgain += t3 - t0;
#endif

    return -gl_cost_of_opening_x;
}


/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */

float
pFL(Points *points, int *feasible, int numfeasible, float z, long *k,
    float cost, long iter, float e)
{
    float change = cost;

    /* continue until we run iter iterations without improvement */
    /* stop instead if improvement is less than e */
    while (change/cost > e) {
        change = 0.0;
        /* randomize order in which centers are considered */

        intshuffle(feasible, numfeasible);

        for (int i = 0; i < iter; i++) {
            int x = i % numfeasible;
            change += pgain(feasible[x], points, z, k);
        }

        cost -= change;
    }
    return cost;
}

int
selectfeasible_fast(Points *points, int **feasible, int kmin)
{
#ifdef TIMING
    double t1 = gettime();
    iters_select_feasible++;
#endif

    int numfeasible = points->num;
    if (numfeasible > (ITER * kmin * log((float)kmin)))
        numfeasible = (int)(ITER * kmin * log((float)kmin));
    *feasible = (int*) malloc(numfeasible * sizeof(int));

    /* not many points, all will be feasible */
    if (numfeasible == points->num) {
        for (int i = 0; i < numfeasible; i++)
            (*feasible)[i] = i;
        return numfeasible;
    }

    float* accumweight = (float*) malloc(points->num * sizeof(float));
    float totalweight;

    accumweight[0] = points->p[0].weight;
    for (int i = 1; i < points->num; i++) {
        accumweight[i] = accumweight[i-1] + points->p[i].weight;
    }
    totalweight = accumweight[points->num-1];

    for (int i = 0; i < numfeasible; i++) {
        float w = (lrand48()/(float)INT_MAX) * totalweight;
        //binary search
        int l = 0;
        int r = points->num - 1;
        if (accumweight[0] > w) {
            (*feasible)[i]=0;
            continue;
        }
        while (l+1 < r) {
            int k = (l+r) / 2;
            if (accumweight[k] > w) {
                r = k;
            } else {
                l = k;
            }
        }
        (*feasible)[i]=r;
    }

    free(accumweight);

#ifdef TIMING
    double t2 = gettime();
    time_select_feasible += t2-t1;
#endif
    return numfeasible;
}

/* compute approximate kmedian on the points */
float
pkmedian(Points *points, long kmin, long kmax, long* kfinal)
{
    float cost;

    long k = 0;
    float hiz = 0.0;

    for (long kk = 0; kk < points->num; kk++) {
        hiz += dist(points->p[kk], points->p[0], points->dim) *
                 points->p[kk].weight;
    }

    if (points->num <= kmax) {
        /* just return all points as facilities */
        for (long kk = 0; kk < points->num; kk++) {
            points->p[kk].assign = kk;
            points->p[kk].cost = 0;
        }
        cost = 0;
        *kfinal = k;
        return cost;
    }

    float loz = 0.0;
    float z = (hiz + loz) / 2.0;
    shuffle(points);
    cost = pspeedy(points, z, &k);

    int i = 0;
    /* give speedy SP chances to get at least kmin/2 facilities */
    while ((k < kmin) && (i < SP)) {
        cost = pspeedy(points, z, &k);
        i++;
    }

    /* if still not enough facilities, assume z is too high */
    while (k < kmin) {
        if (i >= SP) {
            hiz = z;
            z = (hiz + loz) / 2.0;
            i = 0;
        }
        shuffle(points);
        cost = pspeedy(points, z, &k);
        i++;
    }

    /* now we begin the binary search for real */
    /* must designate some points as feasible centers */
    /* this creates more consistency between FL runs */
    /* helps to guarantee correct # of centers at the end */

    int *feasible;
    int numfeasible = selectfeasible_fast(points, &feasible, kmin);
    for (int i = 0; i < points->num; i++) {
        is_center[points->p[i].assign] = true;
    }

    while(1) {
        /* first get a rough estimate on the FL solution */
        cost = pFL(points, feasible, numfeasible, z, &k, cost,
                   (long)(ITER * kmax * log((float)kmax)), 0.1);

        /* if number of centers seems good, try a more accurate FL */
        if (((k <= 1.1*kmax) && (k >= 0.9*kmin)) || ((k <= kmax+2) && (k >= kmin-2))) {
            /* may need to run a little longer here before halting without
            improvement */
            cost = pFL(points, feasible, numfeasible, z, &k, cost,
                       (long)(ITER * kmax * log((float)kmax)), 0.001);
        }

        if (k > kmax) {
            /* facilities too cheap */
            /* increase facility cost and up the cost accordingly */
            loz = z;
            z = (hiz + loz) / 2.0;
            cost += (z - loz) * k;
        }
        if (k < kmin) {
            /* facilities too expensive */
            /* decrease facility cost and reduce the cost accordingly */
            hiz = z;
            z = (hiz + loz) / 2.0;
            cost += (z - hiz) * k;
        }

        /* if k is good, return the result */
        /* if we're stuck, just give up and return what we have */
        if (((k <= kmax) && (k >= kmin)) || (loz >= 0.999 * hiz)) {
            break;
        }
    }

    //clean up...
    free(feasible); 
    *kfinal = k;

    return cost;
}

/* compute the means for the k clusters */
void
contcenters(Points *points)
{
    for (int i = 0; i < points->num; i++) {
        /* compute relative weight of this point to the cluster */
        if (points->p[i].assign != i) {
            float relweight = points->p[points->p[i].assign].weight + points->p[i].weight;
            relweight = points->p[i].weight / relweight;
            for (int ii = 0; ii < points->dim; ii++) {
                points->p[points->p[i].assign].coord[ii] *= 1.0 - relweight;
                points->p[points->p[i].assign].coord[ii] +=
                        points->p[i].coord[ii] * relweight;
            }
            points->p[points->p[i].assign].weight += points->p[i].weight;
        }
    }
}

/* copy centers from points to centers */
void
copycenters(Points *points, Points *centers, long* centerIDs, long offset)
{
    bool *is_a_median = (bool*) malloc(points->num * sizeof(bool));
    memset(is_a_median, 0, points->num * sizeof(bool));

    /* mark the centers */
    for (int i = 0; i < points->num; i++) {
        is_a_median[points->p[i].assign] = true;
    }

    long k = centers->num;

    /* count how many  */
    for (int i = 0; i < points->num; i++) {
        if (is_a_median[i]) {
            memcpy(centers->p[k].coord, points->p[i].coord, points->dim * sizeof(float));
            centers->p[k].weight = points->p[i].weight;
            centerIDs[k] = i + offset;
            k++;
        }
    }

    centers->num = k;

    free(is_a_median);
}

void
localSearch(Points* points, long kmin, long kmax, long* kfinal)
{
#ifdef TIMING
    double t1 = gettime();
#endif

    memset(is_center, 0, points->num * sizeof(bool));
    pkmedian(points, kmin, kmax, kfinal);

#ifdef TIMING
    double t2 = gettime();
    time_local_search += t2-t1;
    iters_local_search++;
#endif
}

class PStream {
  public:
    virtual size_t read( float* dest, int dim, int num ) = 0;
    virtual int ferror() = 0;
    virtual int feof() = 0;
    virtual ~PStream() {}
};

//synthetic stream
class SimStream : public PStream {
  public:
    SimStream(long n_) { n = n_; }

    size_t read(float* dest, int dim, int num) {
        size_t count = 0;
        for (int i = 0; i < num && n > 0; i++) {
            for (int k = 0; k < dim; k++) {
                dest[i * dim + k] = lrand48() / (float)INT_MAX;
            }
            n--;
            count++;
        }
        return count;
    }

    int ferror() { return 0; }
    int feof() { return n <= 0; }
    ~SimStream() {}

  private:
    long n;
};

class FileStream : public PStream {
  public:
    FileStream(char* filename) {
        fp = fopen(filename, "rb");
        if (fp == NULL) {
            fprintf(stderr, "error opening file %s\n.", filename);
            exit(1);
        }
    }
    size_t read( float* dest, int dim, int num ) {
        return std::fread(dest, sizeof(float)*dim, num, fp);
    }
    int ferror() { return std::ferror(fp); }
    int feof() { return std::feof(fp); }
    ~FileStream() {
        printf("closing file stream\n");
        fclose(fp);
    }

  private:
    FILE* fp;
};

void
outcenterIDs(Points* centers, long* centerIDs, char* outfile)
{
    FILE* fp = fopen(outfile, "w");
    if (fp==NULL) {
        fprintf(stderr, "error opening %s\n", outfile);
        exit(1);
    }
    int* is_a_median = (int*) malloc(centers->num * sizeof(int));
    for (int i = 0; i < centers->num; i++) {
        is_a_median[centers->p[i].assign] = 1;
    }

    for (int i = 0; i < centers->num; i++) {
        if (is_a_median[i]) {
            fprintf(fp, "%ld\n", centerIDs[i]);
            fprintf(fp, "%lf\n", centers->p[i].weight);
            for (int k = 0; k < centers->dim; k++) {
                fprintf(fp, "%lf ", centers->p[i].coord[k]);
            }
            fprintf(fp,"\n\n");
        }
    }
    fclose(fp);
    free(is_a_median);
}

void
streamCluster(PStream* stream, long kmin, long kmax, int dim, long chunksize,
    long centersize, char* outfile)
{
    block = (float*) malloc(chunksize * dim * sizeof(float));
    float* centerBlock = (float*) malloc(centersize * dim * sizeof(float));
    long* centerIDs = (long*) malloc(centersize * dim * sizeof(long));

    if (block == NULL) {
        fprintf(stderr, "not enough memory for a chunk!\n");
        exit(1);
    }

    Points points;
    points.dim = dim;
    points.num = chunksize;
    points.p = (Point*) malloc(chunksize * sizeof(Point));
    for (int i = 0; i < chunksize; i++) {
        points.p[i].coord = &block[i * dim];
    }

    switch_membership = (bool*) malloc(points.num * sizeof(bool));
    is_center = (bool*) malloc(points.num * sizeof(bool));
    center_table = (int*) malloc(points.num * sizeof(int));

    Points centers;
    centers.dim = dim;
    centers.p = (Point*) malloc(centersize * sizeof(Point));
    centers.num = 0;

    for (int i = 0; i < centersize; i++) {
        centers.p[i].coord = &centerBlock[i * dim];
        centers.p[i].weight = 1.0;
    }

    long IDoffset = 0;
    long kfinal;
    while(1) {
        size_t numRead = stream->read(block, dim, chunksize);
        printf("read %u points\n", (unsigned)numRead);

        if (stream->ferror() || (numRead < (unsigned int)chunksize && !stream->feof())) {
            fprintf(stderr, "error reading data!\n");
            exit(1);
        }

        points.num = numRead;
        for (int i = 0; i < points.num; i++) {
            points.p[i].weight = 1.0;
        }

        localSearch(&points, kmin, kmax, &kfinal);

        printf("finish local search\n");
        if (kfinal + centers.num > centersize) {
            //here we don't handle the situation where # of centers gets too large.
            fprintf(stderr, "oops! no more space for centers\n");
            exit(1);
        }

        contcenters(&points);
        copycenters(&points, &centers, centerIDs, IDoffset);
        IDoffset += numRead;

        if (stream->feof()) {
            break;
        }
    }

    //finally cluster all temp centers
    localSearch(&centers, kmin, kmax, &kfinal);

    contcenters(&centers);
    outcenterIDs(&centers, centerIDs, outfile);

    free(switch_membership);
    free(is_center);
    free(center_table);

    free(block);
    free(centerBlock);
    free(centerIDs);
    free(points.p);
    free(centers.p);
}

int
main(int argc, char **argv)
{
    char *outfilename = new char[MAXNAMESIZE];
    char *infilename = new char[MAXNAMESIZE];
    long kmin, kmax, n, chunksize, clustersize;
    int dim;

#ifdef TIMING
    iters_local_search = 0;
    iters_speedy = 0;
    iters_shuffle = 0;
    iters_select_feasible = 0;
    iters_pgain = 0;
#endif

    printf("PARSEC Benchmark Suite\n");
    fflush(NULL);

    if (argc < 10) {
        fprintf(stderr, "usage: %s k1 k2 d n chunksize clustersize infile outfile nproc\n", argv[0]);
        fprintf(stderr, "  k1:          Min. number of centers allowed\n");
        fprintf(stderr, "  k2:          Max. number of centers allowed\n");
        fprintf(stderr, "  d:           Dimension of each data point\n");
        fprintf(stderr, "  n:           Number of data points\n");
        fprintf(stderr, "  chunksize:   Number of data points to handle per step\n");
        fprintf(stderr, "  clustersize: Maximum number of intermediate centers\n");
        fprintf(stderr, "  infile:      Input file (if n<=0)\n");
        fprintf(stderr, "  outfile:     Output file\n");
        fprintf(stderr, "  nproc:       Number of threads to use\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "if n > 0, points will be randomly generated instead of reading from infile.\n");
        exit(1);
    }
    kmin = atoi(argv[1]);
    kmax = atoi(argv[2]);
    dim = atoi(argv[3]);
    n = atoi(argv[4]);
    chunksize = atoi(argv[5]);
    clustersize = atoi(argv[6]);
    strcpy(infilename, argv[7]);
    strcpy(outfilename, argv[8]);

    ompthreads = atoi(argv[9]);
    omp_set_num_threads(ompthreads);

    srand48(SEED);
    PStream* stream;
    if (n > 0) {
        stream = new SimStream(n);
    } else {
        stream = new FileStream(infilename);
    }

#ifdef TIMING
    double t1 = gettime();
#endif

#ifdef GEM5_WORK
    m5_work_begin(0, 0);
    m5_dumpreset_stats(0, 0);
#endif

    streamCluster(stream, kmin, kmax, dim, chunksize, clustersize, outfilename);

#ifdef GEM5_WORK
    m5_dumpreset_stats(0, 0);
    m5_work_end(0, 0);
#endif

    delete [] infilename;
    delete [] outfilename;
    delete stream;

#ifdef TIMING
    double t2 = gettime();
    double per_local_search = time_local_search / iters_local_search;
    double per_speedy = time_speedy / iters_speedy;
    double per_shuffle = time_shuffle / iters_shuffle;
    double per_select_feasible = time_select_feasible / iters_select_feasible;
    double per_pgain = time_pgain / iters_pgain;
    double per_pgain_init = time_pgain_init / iters_pgain;
    double per_pgain_dist = time_pgain_dist / iters_pgain;
    double per_pgain_rest = (time_pgain - time_pgain_init - time_pgain_dist) / iters_pgain;
    printf("ROI Runtime: %lf\n", t2 - t1);
    printf("  localSearch: time: %lf, calls: %d, per: %lf\n", time_local_search, iters_local_search, per_local_search);
    printf("    pspeedy: time: %lf, calls: %d, per: %lf\n", time_speedy, iters_speedy, per_speedy);
    printf("    pshuffle: time: %lf, calls: %d, per: %lf\n", time_shuffle, iters_shuffle, per_shuffle);
    printf("    pselect: time: %lf, calls: %d, per: %lf\n", time_select_feasible, iters_select_feasible, per_select_feasible);
    printf("    pgain: time: %lf, calls: %d, per: %lf\n", time_pgain, iters_pgain, per_pgain);
    printf("      pgain_init: time: %lf, per: %lf\n", time_pgain_init, per_pgain_init);
    printf("      pgain_dist: time: %lf, per: %lf\n", time_pgain_dist, per_pgain_dist);
    printf("      pgain_rest: time: %lf, per: %lf\n", time_pgain - time_pgain_init - time_pgain_dist, per_pgain_rest);
#endif

    return 0;
}
