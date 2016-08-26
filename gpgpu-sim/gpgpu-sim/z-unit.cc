//// Copyright (c) 2009-2011, Tor M. Aamodt
//// The University of British Columbia
//// All rights reserved.
////
//// Redistribution and use in source and binary forms, with or without
//// modification, are permitted provided that the following conditions are met:
////
//// Redistributions of source code must retain the above copyright notice, this
//// list of conditions and the following disclaimer.
//// Redistributions in binary form must reproduce the above copyright notice, this
//// list of conditions and the following disclaimer in the documentation and/or
//// other materials provided with the distribution.
//// Neither the name of The University of British Columbia nor the names of its
//// contributors may be used to endorse or promote products derived from this
//// software without specific prior written permission.
////
//// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//#include"z-unit.h"
//#include"l2cache.h"
//#include "gpu-sim.h"
//#include "../cuda-sim/memory.h"
//#include "../cuda-sim/ptx_ir.h"
//
//void z_cache::issue_request(unsigned cycle) {
//    unsigned time = cycle;
//    request_buffer_entry* request;
//    bool issued = m_request_buffer.request_scheduler(request);
//    if (!issued)
//        return;
//
//    mem_fetch* mf = request->mf;
//    assert(mf->get_access_type() == Z_ACCESS_TYPE);
//    new_addr_type addr = mf->get_partition_addr();
//    new_addr_type block_addr = m_config.block_addr(addr);
//    unsigned cache_index = (unsigned) - 1;
//    enum cache_request_status status = m_tag_array.probe(block_addr, cache_index);
//
//    if (status == HIT) {
//        if (!m_z_check_buffer.full()) {
//            z_check_buffer_entry *new_entry = m_z_check_buffer.allocate(mf, block_addr, cache_index, mf->get_data_size());
//            new_entry->m_ready = true;
//            new_entry->m_request_buffer_entry = request;
//            m_tag_array.access(block_addr, time, cache_index);
//            request->index = cache_index;
//            request->status = ISSUED;
//            return;
//        }
//    }
//    if (status == MISS) {
//        if (m_miss_queue.size() >= m_miss_queue_size - 1) {
//            return; //not schedule any thing
//        }
//        if (!m_z_check_buffer.full()) {
//            m_miss_queue.push_back(mf);
//            z_check_buffer_entry *new_entry = m_z_check_buffer.allocate(mf, block_addr, cache_index, mf->get_data_size());
//            new_entry->m_request_buffer_entry = request;
//            request->index = cache_index;
//            request->status = ISSUED;
//            bool wb = false;
//            cache_block_t evicted;
//            m_tag_array.access(block_addr, time, cache_index, wb, evicted);
//            if (wb) {
//                mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr, Z_UNIT_WRBK_ACC, m_config.get_line_sz(), true);
//                m_miss_queue.push_back(wb);
//            }
//        }
//        return;
//    }
//    if (status == HIT_RESERVED) {
//        request->status = BLOCKED;
//        request->index = cache_index;
//    }
//}
//
//void z_cache::miss_queue_cycle() {
//    if (!m_miss_queue.empty()) {
//        mem_fetch *mf = m_miss_queue.front();
//        if (!m_memport->full(mf->get_data_size(), mf->get_is_write())) {
//            m_miss_queue.pop_front();
//            m_memport->push(mf);
//        }
//    }
//}
//
//void z_cache::release_request_entry(mem_fetch* mf) {
//    z_unit_interface* reply_port = dynamic_cast<z_unit_interface*> (m_memport);
//    int index = m_z_check_buffer.get_entry_number_for_C_request(mf);
//    z_check_buffer_entry* entry = m_z_check_buffer.get_entry(index);
//    mem_fetch* zmf = entry->mf;
//    //zmf->set_new_address(entry->m_request_buffer_entry->back_up_C_address,true,m_memory_config);
//    zmf->set_reply();
//    reply_port->reply_queue_push(zmf);
//    m_request_buffer.release_entry(entry->m_request_buffer_entry);
//    m_z_check_buffer.release_entry(index);
//}
//
//void z_cache::z_checker(unsigned cycle) {
//    if (m_miss_queue.size() >= m_miss_queue_size)
//        return;
//    z_unit_interface* reply_port = dynamic_cast<z_unit_interface*> (m_memport);
//    assert(reply_port);
//    if (reply_port->reply_queue_full())
//        return;
//    unsigned time = cycle;
//    int selected_entry = m_z_check_buffer.get_next_entry_number();
//    if (selected_entry == -1)
//        return;
//    z_check_buffer_entry* entry = m_z_check_buffer.element(selected_entry);
//    mem_fetch* mf = entry->mf;
//    mf->set_new_address(entry->m_request_buffer_entry->back_up_C_address, true, m_memory_config);
//    mf->set_atomic(true);
//    assert(entry->m_valid && entry->m_ready);
//    assert(mf);
//    active_mask_t access_mask = mf->do_zrop();
//    ///TODO make sure (int)access_mask.size() is true
//    bool update = false;
//    int count_updates = 0;
//    for (unsigned i = 0; i < access_mask.size(); i++) {
//        if (access_mask.test(i)) {
//            update = true;
//            count_updates++;
//            //break;
//        }
//    }
//    if (update and !isBlendingEnabled()) {
//        m_color_writes_depth_test++;
//        mem_fetch *newmf = m_memfetch_creator->alloc(mf->get_addr(), Z_UNIT_C_UPDATE, access_mask.size() * C_DATA_SIZE, true); //zzzz
//        entry->color_generated_mf = newmf;
//        m_miss_queue.push_back(newmf);
//        cache_block_t &block = m_tag_array.get_block(entry->m_cache_index);
//        // prevent this entry from re-scheduling (Already scheduled)
//        entry->m_ready = false;
//        m_request_buffer.release_next_waiting_entry(entry->m_request_buffer_entry);
//        m_tag_array.fill(entry->m_cache_index, time);
//        block.dirty = true;
//    } else {
//        if(isBlendingEnabled()){
//            assert(update);
//            m_color_writes_blending++;
//            cache_block_t &block = m_tag_array.get_block(entry->m_cache_index);
//            block.dirty = true;
//        }
//        mf->set_reply();
//        reply_port->reply_queue_push(mf);
//        m_request_buffer.release_entry(entry->m_request_buffer_entry);
//        m_request_buffer.release_next_waiting_entry(entry->m_request_buffer_entry);
//        m_tag_array.fill(entry->m_cache_index, time);
//        m_z_check_buffer.release_entry(selected_entry);
//    }
//}
//
//void z_cache::mark_z_access_ready(mem_fetch* mf) {
//    int index = m_z_check_buffer.get_entry_number(mf);
//    m_z_check_buffer.mark_ready(index);
//}
//
//bool z_cache::insert_into_incoming_buffer(mem_fetch* mf) {
//    bool status = m_request_buffer.insert_request(mf);
//    if (status) {
//        new_addr_type addr = mf->get_addr();
//        // Flip 37 bit of c address to get z address (This is just a magic number
//        //if blending then we want to fetch the color not the depth so we keep the address the same as the color address
//        if (!isBlendingEnabled())
//            addr ^= 0x0000001000000000; //zzzz make it safe to add as much as many memory partitions
//        mf->set_new_address(addr, false, m_memory_config);
//        mf->set_atomic(false);
//    }
//    return status;
//}
//
//void z_cache::cycle(unsigned cycle) {
//    z_checker(cycle);
//    miss_queue_cycle();
//    issue_request(cycle);
//}
//
//bool z_cache::request_buffer::request_scheduler(request_buffer_entry*& req) {
//    int selected_entry = -1;
//    unsigned time = (unsigned) - 1;
//    for (int i = 0; i < (int) size; i++) {
//        if (m_request_buffer[i].status != VALID)
//            continue;
//        if (m_request_buffer[i].arriving_time < time) {
//            time = m_request_buffer[i].arriving_time;
//            selected_entry = i;
//        }
//    }
//    if (selected_entry == -1)
//        return false;
//    req = &m_request_buffer[selected_entry];
//    assert(req->mf != NULL);
//    return true;
//}
//
//bool z_cache::request_buffer::insert_request(mem_fetch* mf) {
//    int selected_entry = -1;
//    for (unsigned int i = 0; i < size; i++) {
//        if (m_request_buffer[i].status == INVALID) {
//            selected_entry = (int) i;
//            break;
//        }
//    }
//    if (selected_entry == -1)
//        abort();//return false; the return value not used!!
//    assert(m_request_buffer[selected_entry].status == INVALID);
//    m_request_buffer[selected_entry].mf = mf;
//    m_request_buffer[selected_entry].status = VALID;
//    m_request_buffer[selected_entry].arriving_time = time;
//    m_request_buffer[selected_entry].back_up_C_address = mf->get_addr();
//    time++;
//    return true;
//}
//
//void z_cache::request_buffer::release_entry(request_buffer_entry* entry) {
//    entry->status = INVALID;
//}
//
//bool z_cache::request_buffer::full() {
//    for (int i = 0; i < (int) size; i++) {
//        if (m_request_buffer[i].status == INVALID)
//            return false;
//    }
//    return true;
//}
//
//void z_cache::request_buffer::release_next_waiting_entry(request_buffer_entry* entry) {
//    int index = entry->index;
//    unsigned min = (unsigned) - 1;
//    int next_entry = -1;
//    for (int i = 0; i < (int) size; i++) {
//        if ((int) m_request_buffer[i].index == index && m_request_buffer[i].status == BLOCKED) {
//            if (m_request_buffer[i].arriving_time < min) {
//                min = m_request_buffer[i].arriving_time;
//                next_entry = i;
//            }
//        }
//    }
//    if (next_entry != -1) {
//        m_request_buffer[next_entry].status = VALID;
//    }
//}
//
//void z_cache::request_buffer::dump()const {
//    for (int i = 0; i < (int) size; i++) {
//        switch (m_request_buffer[i].status) {
//            case INVALID:
//                printf("I\t---\t---\n");
//                break;
//            case VALID:
//                printf("V\t%u\t%u\t%u\n", m_request_buffer[i].mf->get_request_uid(), m_request_buffer[i].index, m_request_buffer[i].arriving_time);
//                break;
//            case ISSUED:
//                printf("IS\t%u\t%u\t%u\n", m_request_buffer[i].mf->get_request_uid(), m_request_buffer[i].index, m_request_buffer[i].arriving_time);
//                break;
//            case BLOCKED:
//                printf("B\t%u\t%u\t%u\n", m_request_buffer[i].mf->get_request_uid(), m_request_buffer[i].index, m_request_buffer[i].arriving_time);
//                break;
//        }
//    }
//}
//
//void z_cache::print(FILE *fp, unsigned &accesses, unsigned &misses, unsigned &depthColorWrites, unsigned &blendingColorWrites) const {
//    fprintf(fp, "Z_unit %s:\t", m_name.c_str());
//    m_tag_array.print(fp, accesses, misses);
//    fprintf(fp, "Color write requests sent to memory, On depth test= %u On blending = %u\n", m_color_writes_depth_test, m_color_writes_blending);
//    depthColorWrites += m_color_writes_depth_test;
//    blendingColorWrites += m_color_writes_blending;
//}
//
//enum cache_request_status z_tag_array::access(new_addr_type addr, unsigned time, unsigned &idx) {
//    bool wb = false;
//    cache_block_t evicted;
//    enum cache_request_status result = access(addr, time, idx, wb, evicted);
//    assert(!wb);
//    return result;
//}
//
//enum cache_request_status z_tag_array::access(new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted) {
//    m_access++;
//    //shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
//    enum cache_request_status status = probe(addr, idx);
//    switch (status) {
//        case HIT_RESERVED:
//            m_pending_hit++;
//        case HIT:
//            m_lines[idx].m_last_access_time = time;
//            m_lines[idx].m_status = RESERVED;
//            break;
//        case MISS:
//            m_miss++;
//            //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
//            if (m_config.m_alloc_policy == ON_MISS) {
//                if (m_lines[idx].dirty) {
//                    wb = true;
//                    evicted = m_lines[idx];
//                }
//                m_lines[idx].allocate(m_config.tag(addr), m_config.block_addr(addr), time);
//            }
//            break;
//        case RESERVATION_FAIL:
//            m_miss++;
//            //		shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
//            break;
//    }
//    assert(m_lines[idx].m_status != MODIFIED);
//    return status;
//}
//
//z_cache::z_cache(const char *name, const memory_config *config,
//        mem_fetch_interface* mf_interface, mem_fetch_allocator* mf_allocator)
//        :baseline_cache( &(*name), config->m_z_unit_config.m_cache_config,
//                    -1, -1, &(*mf_interface), IN_ZUNIT_CACHE, (tag_array*)&m_tag_array ),
//        m_config(config->m_z_unit_config.m_cache_config),
//        m_tag_array(config->m_z_unit_config.m_cache_config),
//        m_request_buffer(config->m_z_unit_config.m_request_buffer_size),
//        m_z_check_buffer(config->m_z_unit_config.m_z_checker_buffer_size) {
//    m_miss_queue_size = config->m_z_unit_config.m_miss_queue_size;
//    m_request_buffer.m_z_cache = this;
//    m_memfetch_creator = mf_allocator;
//    m_memport = mf_interface;
//    m_memory_config = config;
//    m_name = name;
//    m_color_writes_blending =0;
//    m_color_writes_depth_test =0;
//}
//
//z_unit::z_unit(const char * name, const memory_config * mem_config, mem_fetch_interface* mf_interface,
//        mem_fetch_allocator* mf_allocator) {
//    m_memory_config = mem_config;
//    m_z_cache = new z_cache(name, mem_config, mf_interface, mf_allocator);
//}
//
//void z_cache::z_check_buffer::dump() const {
//
//}
