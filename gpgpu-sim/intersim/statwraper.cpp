//a Wraper function for stats class
#include "stats.hpp"
#include <stdio.h>

Stats_gpgpu* StatCreate (const char * name, double bin_size, int num_bins) {
   Stats_gpgpu* newstat = new Stats_gpgpu(NULL,name,bin_size,num_bins);
   newstat->Clear ();
   return newstat;  
}

void StatClear(void * st)
{
   ((Stats_gpgpu *)st)->Clear();
}

void StatAddSample (void * st, int val)
{
   ((Stats_gpgpu *)st)->AddSample(val);
}

double StatAverage(void * st) 
{
   return((Stats_gpgpu *)st)->Average();
}

double StatMax(void * st) 
{
   return((Stats_gpgpu *)st)->Max();
}

double StatMin(void * st) 
{
   return((Stats_gpgpu *)st)->Min();
}

void StatDisp (void * st)
{
   printf ("Stats for ");
   ((Stats_gpgpu *)st)->DisplayHierarchy();
   if (((Stats_gpgpu *)st)->NeverUsed()) {
      printf (" was never updated!\n");
   } else {
      printf("Min %f Max %f Average %f \n",((Stats_gpgpu *)st)->Min(),((Stats_gpgpu *)st)->Max(),StatAverage(st));
      ((Stats_gpgpu *)st)->Display();
   }
}

#if 0 
int main ()
{
   void * mytest = StatCreate("Test",1,5);
   StatAddSample(mytest,4);
   StatAddSample(mytest,4);StatAddSample(mytest,4);
   StatAddSample(mytest,2);
   StatDisp(mytest);
}
#endif


