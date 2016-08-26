#ifndef _STATS_GPGPU_HPP_
#define _STATS_GPGPU_HPP_

#include "module.hpp"

class Stats_gpgpu : public Module {
   int    _num_samples;
   double _sample_sum;

   bool _reset;
   double _min;
   double _max;

   int    _num_bins;
   double _bin_size;

   int *_hist;

public:
   Stats_gpgpu( Module *parent, const string &name,
          double bin_size = 1.0, int num_bins = 10 );
   ~Stats_gpgpu( );

   void Clear( );

   double Average( ) const;
   double Max( ) const;
   double Min( ) const;
   int    NumSamples( ) const;

   void AddSample( double val );
   void AddSample( int val );


   void Display( ) const;
   bool NeverUsed() const;
};

#endif
