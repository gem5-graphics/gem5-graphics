#ifndef GPU_CACHE_GEM5_H_
#define GPU_CACHE_GEM5_H_

#include "gpu-cache.h"
#include "base/misc.hh"
#include "gpu/gpgpu-sim/cuda_core.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

class read_only_cache_gem5 : public read_only_cache {
private:
    gpgpu_t* abstractGPU;
    CudaCore* shaderCore;
    unsigned m_sid;
    const _memory_space_t m_mem_space;
public:
    read_only_cache_gem5(gpgpu_t* _gpu, const char *name,
            cache_config &config, int core_id, int type_id,
            mem_fetch_interface *memport, enum mem_fetch_status status, _memory_space_t mem_space);
    enum cache_request_status access(new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events);
    void cycle();
};


class l1tcache_gem5 : public tex_cache {
    gpgpu_t* abstractGPU;
    CudaCore* shaderCore;
    unsigned m_sid;
public:
    l1tcache_gem5(gpgpu_t* _gpu, const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status request_status, enum mem_fetch_status rob_status);
    enum cache_request_status access(new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events);
    void cycle();
};
#endif /* GPU_CACHE_GEM5_H_ */