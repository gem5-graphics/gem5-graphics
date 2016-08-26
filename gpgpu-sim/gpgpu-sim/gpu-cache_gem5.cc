#include "gpu-cache_gem5.h"

read_only_cache_gem5::read_only_cache_gem5(gpgpu_t* _gpu, const char *name, cache_config &config,
        int core_id, int type_id, mem_fetch_interface *memport,
        enum mem_fetch_status status, _memory_space_t mem_space)
    : read_only_cache(name, config, core_id, type_id, memport, status),
      abstractGPU(_gpu), shaderCore(NULL), m_mem_space(mem_space)
{
    m_sid = core_id;
}

enum cache_request_status
read_only_cache_gem5::access(new_addr_type addr, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events)
{
    assert( mf->get_data_size() <= m_config.get_line_sz());
    assert(m_config.m_write_policy == READ_ONLY);
    assert(!mf->get_is_write());
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
    if ( status == HIT ) {
        m_tag_array->access(block_addr,time,cache_index); // update LRU state
        return HIT;
    }
    if ( status != RESERVATION_FAIL ) {
        bool mshr_hit = m_mshrs.probe(block_addr);
        bool mshr_avail = !m_mshrs.full(block_addr);
        if ( mshr_hit && mshr_avail ) {
            m_tag_array->access(addr,time,cache_index);
            m_mshrs.add(block_addr,mf);
            return MISS;
        } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
            m_tag_array->access(addr,time,cache_index);
            m_mshrs.add(block_addr,mf);
            m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
            // @TODO: Can we move this so that it isn't executed each call?
            if (!shaderCore) {
                shaderCore = abstractGPU->gem5CudaGPU->getCudaCore(m_sid);
            }
            // Send access into Ruby through shader core
            mf->set_data_size( m_config.get_line_sz());
            switch(m_mem_space){
                case instruction_space: shaderCore->icacheFetch((Addr)addr, mf) ; break;
                //case tex_space: shaderCore->texCacheFetch((Addr)addr, mf); break;
                case tex_space: panic("Shouldn't get a tex fetch here\n"); break;
                default: 
                    printf ("GPGPU-Sim uArch: ERROR unexpected memory space type for read_only_cache_gem5\n");
                    abort();
            }
            mf->set_status(m_miss_queue_status,time);
            events.push_back(READ_REQUEST_SENT);
            return MISS;
        }
    }
    return RESERVATION_FAIL;
}

void read_only_cache_gem5::cycle()
{
    bool data_port_busy = !m_bandwidth_management.data_port_free(); 
    bool fill_port_busy = !m_bandwidth_management.fill_port_free(); 
    m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
    m_bandwidth_management.replenish_port_bandwidth();
}

l1tcache_gem5::l1tcache_gem5(gpgpu_t* _gpu, const char* name, cache_config& config, int core_id, int type_id, mem_fetch_interface* memport,
        mem_fetch_status request_status, mem_fetch_status rob_status)
        : tex_cache(name,config,core_id, type_id, memport, request_status, rob_status),
        abstractGPU(_gpu), shaderCore(NULL)
{
    m_sid = core_id;
}

enum cache_request_status
l1tcache_gem5::access(new_addr_type addr, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events) {
    if (m_fragment_fifo.full() || m_rob.full())
        return RESERVATION_FAIL;

    if (!shaderCore)
        shaderCore = abstractGPU->gem5CudaGPU->getCudaCore(m_sid);
    
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status = m_tags.probe(block_addr,cache_index);
    assert( probe_status != RESERVATION_FAIL );
    assert( probe_status != HIT_RESERVED );
    if(probe_status==MISS and !shaderCore->texCacheResAvailabe((Addr)addr))
        return RESERVATION_FAIL;
    
    
    assert( mf->get_data_size() == m_config.get_line_sz());
    // at this point, we will accept the request : access tags and immediately allocate line
    enum cache_request_status status = m_tags.access(block_addr,time,cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;
    assert( status != RESERVATION_FAIL );
    assert( status != HIT_RESERVED ); // as far as tags are concerned: HIT or MISS
    m_fragment_fifo.push( fragment_entry(mf,cache_index,status==MISS,mf->get_data_size()) );
    if ( status == MISS ) {
        // we need to send a memory request...
        unsigned rob_index = m_rob.push( rob_entry(cache_index, mf, block_addr) );
        m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
        m_tags.fill(cache_index,time); // mark block as valid

        // Send access into Ruby through shader core
        shaderCore->texCacheFetch((Addr)addr, mf);
        mf->set_status(m_request_queue_status,time);
        events.push_back(READ_REQUEST_SENT);
        cache_status = MISS;
    } else {
        // the value *will* *be* in the cache already
        cache_status = HIT_RESERVED;
    }
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}


void l1tcache_gem5::cycle()
{
    read_ready_lines();
}
