#include <stdio.h>
#include <omp.h>

extern int omp_num_threads;

void lud_omp(float *a, int size)
{
     int i,j,k;
     float sum;
	 printf("num of threads = %d\n", omp_num_threads);
     for (i=0; i <size; i++){
		omp_set_num_threads(omp_num_threads);
#pragma omp parallel for default(none) \
         private(j,k,sum) shared(size,i,a) 
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }
#pragma omp parallel for default(none) \
         private(j,k,sum) shared(size,i,a) 
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}
