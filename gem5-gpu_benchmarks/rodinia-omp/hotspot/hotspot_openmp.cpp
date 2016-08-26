#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;

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

#define STR_SIZE    256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)    */
#define MAX_PD    (3.0e6)
/* required precision in degrees    */
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor    */
#define FACTOR_CHIP    0.5
#define OPEN
//#define NUM_THREAD 4

/* chip parameters    */
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all    */
float amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration(float *result, float *temp, float *power, int row, int col,
                      float Cap, float Rx, float Ry, float Rz,
                      float step)
{
    float delta;
    int r, c;
    float step_div_Cap = step / Cap;
    float Rx_inv = 1 / Rx;
    float Ry_inv = 1 / Ry;
    float Rz_inv = 1 / Rz;
    //printf("num_omp_threads: %d\n", num_omp_threads);
#ifdef OPEN
    omp_set_num_threads(num_omp_threads);
    #pragma omp parallel for shared(power, temp,result) private(r, c, delta) firstprivate(row, col) schedule(static)
#endif

    for (r = 0; r < row; r++) {
        int curr_row = r*col;
        for (c = 0; c < col; c++) {
              /*    Corner 1    */
            if ( (r == 0) && (c == 0) ) {
                delta = step_div_Cap * (power[0] +
                        (temp[1] - temp[0]) * Rx_inv +
                        (temp[col] - temp[0]) * Ry_inv +
                        (amb_temp - temp[0]) * Rz_inv);
            }    /*    Corner 2    */
            else if ((r == 0) && (c == col-1)) {
                delta = step_div_Cap * (power[c] +
                        (temp[c-1] - temp[c]) * Rx_inv +
                        (temp[c+col] - temp[c]) * Ry_inv +
                        (amb_temp - temp[c]) * Rz_inv);
            }    /*    Corner 3    */
            else if ((r == row-1) && (c == col-1)) {
                delta = step_div_Cap * (power[curr_row+c] +
                        (temp[curr_row+c-1] - temp[curr_row+c]) * Rx_inv +
                        (temp[curr_row-col+c] - temp[curr_row+c]) * Ry_inv +
                        (amb_temp - temp[curr_row+c]) * Rz_inv);
            }    /*    Corner 4    */
            else if ((r == row-1) && (c == 0)) {
                delta = step_div_Cap * (power[curr_row] +
                        (temp[curr_row+1] - temp[curr_row]) * Rx_inv +
                        (temp[curr_row-col] - temp[curr_row]) * Ry_inv +
                        (amb_temp - temp[curr_row]) * Rz_inv);
            }    /*    Edge 1    */
            else if (r == 0) {
                delta = step_div_Cap * (power[c] +
                        (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_inv +
                        (temp[col+c] - temp[c]) * Ry_inv +
                        (amb_temp - temp[c]) * Rz_inv);
            }    /*    Edge 2    */
            else if (c == col-1) {
                delta = step_div_Cap * (power[curr_row+c] +
                        (temp[curr_row+col+c] + temp[curr_row-col+c] - 2.0*temp[curr_row+c]) * Ry_inv +
                        (temp[curr_row+c-1] - temp[curr_row+c]) * Rx_inv +
                        (amb_temp - temp[curr_row+c]) * Rz_inv);
            }    /*    Edge 3    */
            else if (r == row-1) {
                delta = step_div_Cap * (power[curr_row+c] +
                        (temp[curr_row+c+1] + temp[curr_row+c-1] - 2.0*temp[curr_row+c]) * Rx_inv +
                        (temp[curr_row-col+c] - temp[curr_row+c]) * Ry_inv +
                        (amb_temp - temp[curr_row+c]) * Rz_inv);
            }    /*    Edge 4    */
            else if (c == 0) {
                delta = step_div_Cap * (power[curr_row] +
                        (temp[curr_row+col] + temp[curr_row-col] - 2.0*temp[curr_row]) * Ry_inv +
                        (temp[curr_row+1] - temp[curr_row]) * Rx_inv +
                        (amb_temp - temp[curr_row]) * Rz_inv);
            }    /*    Inside the chip    */
            else {
                delta = step_div_Cap * (power[curr_row+c] +
                        (temp[curr_row+col+c] + temp[curr_row-col+c] - 2.0*temp[curr_row+c]) * Ry_inv +
                        (temp[curr_row+c+1] + temp[curr_row+c-1] - 2.0*temp[curr_row+c]) * Rx_inv +
                        (amb_temp - temp[curr_row+c]) * Rz_inv);
            }

            /*    Update Temperatures    */
            result[curr_row+c] =temp[curr_row+c]+ delta;


        }
    }
}

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
float *compute_tran_temp(float *result, int num_iterations, float *temp, float *power, int row, int col)
{
    #ifdef VERBOSE
    int i = 0;
    #endif

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;

    #ifdef VERBOSE
    fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
    fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
    #endif

    float *swap_ptr;
    for (int i = 0; i < num_iterations ; i++)
    {
        #ifdef VERBOSE
        fprintf(stdout, "iteration %d\n", i);
        #endif
        single_iteration(result, temp, power, row, col, Cap, Rx, Ry, Rz, step);
        swap_ptr = temp;
        temp = result;
        result = swap_ptr;
    }
    return temp;
}

void fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
    exit(1);
}

void read_input(float *vect, int grid_rows, int grid_cols, char *file)
{
    int i, index;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    fp = fopen (file, "r");
    if (!fp)
        fatal ("file could not be opened for reading");

    for (i=0; i < grid_rows * grid_cols; i++) {
        fgets(str, STR_SIZE, fp);
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1) )
            fatal("invalid file format");
        vect[i] = val;
    }

    fclose(fp);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
    fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<no. of threads>   - number of threads\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    exit(1);
}

int main(int argc, char **argv)
{
    int grid_rows, grid_cols, sim_time, i;
    float *temp, *power, *result;
    char *tfile, *pfile;

    /* check validity of inputs    */
    if (argc != 7)
        usage(argc, argv);
    if ((grid_rows = atoi(argv[1])) <= 0 ||
        (grid_cols = atoi(argv[1])) <= 0 ||
        (sim_time = atoi(argv[3])) <= 0 ||
        (num_omp_threads = atoi(argv[4])) <= 0
        )
        usage(argc, argv);

    /* allocate memory for the temperature and power arrays    */
    temp = (float *) calloc (grid_rows * grid_cols, sizeof(float));
    power = (float *) calloc (grid_rows * grid_cols, sizeof(float));
    result = (float *) calloc (grid_rows * grid_cols, sizeof(float));
    if(!temp || !power)
        fatal("unable to allocate memory");

    /* read initial temperatures and input power    */
    tfile = argv[5];
    pfile = argv[6];
    read_input(temp, grid_rows, grid_cols, tfile);
    read_input(power, grid_rows, grid_cols, pfile);

    printf("Start computing the transient temperature\n");

#ifdef GEM5_WORK
    m5_work_begin(0, 0);
    m5_dumpreset_stats(0, 0);
#endif

#ifdef TIMING
    double start_time = gettime();
#endif

    temp = compute_tran_temp(result, sim_time, temp, power, grid_rows, grid_cols);

#ifdef TIMING
    double end_time = gettime();
    printf("ROI Runtime: %f\n", end_time - start_time);
#endif

#ifdef GEM5_WORK
    m5_dumpreset_stats(0, 0);
    m5_work_end(0, 0);
#endif

    printf("Ending simulation\n");
    /* output results    */
#ifdef VERBOSE
    fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
    for (int i=0; i < grid_rows && i<10; i++) {
        for (int j=0; j < grid_cols && j<50; j++)
        {
            printf("%g ", temp[i*grid_cols+j]);
        }
        printf("\n");
    }
#endif
    /* cleanup    */
    free(temp);
    free(power);

    return 0;
}

