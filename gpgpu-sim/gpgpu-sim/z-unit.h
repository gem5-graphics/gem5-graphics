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
//
//#ifndef Z_UNIT_H
//#define Z_UNIT_H
//
//
//#include "../abstract_hardware_model.h"
//#include "gpu-cache.h"
//#include "delayqueue.h"
//#define WRITE_PACKET_SIZE 8
//class mem_fetch;
//class z_unit_config{
//public:
//	z_unit_config(){
//	}
//	void init(){
//		m_cache_config.init(m_cache_config_string, FuncCachePreferNone);
//	}
//	char *m_cache_config_string;
//	bool disabled() const{
//		return m_disabled;
//	}
//	void reg_options(class OptionParser * opp){
//		option_parser_register(opp, "-gpgpu_cache:z_unit", OPT_CSTR, &m_cache_config_string,
//				"z-unit cache configuration "
//				" {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:**<fifo_entry>}",
//				"32:256:1,L:B:m:N,A:32:8,8");
//		option_parser_register(opp, "-z_unit_disabled", OPT_BOOL, &m_disabled,
//				"disabling z-unit ",
//				"0");
//		option_parser_register(opp, "-z_unit_miss_queue_size", OPT_UINT32, &m_miss_queue_size,
//				"z_unit miss queue size",
//				"8");
//		option_parser_register(opp, "-z_unit_request_buffer_size", OPT_UINT32, &m_request_buffer_size,
//				"number of in fly requests in z_unit",
//				"16");
//		option_parser_register(opp, "-z_unit_z_checker_buffer_size", OPT_UINT32, &m_z_checker_buffer_size,
//				"buffer size for request ready to be checked by z tester unit",
//				"8");
//	}
//	bool m_disabled;
//	cache_config m_cache_config;
//	unsigned m_miss_queue_size;
//	unsigned m_request_buffer_size;
//	unsigned m_z_checker_buffer_size;
//
//private:
//
//};
//
//class z_tag_array: public tag_array{
//public:
//	z_tag_array( const cache_config &config):tag_array(config,-1,-1){}
//	virtual enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx );
//	virtual enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted );
//};
//
//class z_cache : public baseline_cache{
//public:
//	z_cache( const char *name, const memory_config *config,
//			mem_fetch_interface* mf_interface,mem_fetch_allocator* mf_allocator);
//	virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events )
//	{
//		return RESERVATION_FAIL;
//	}
//	bool insert_into_incoming_buffer(mem_fetch* mf);
//	void cycle(unsigned cycle);
//        void print(FILE *fp, unsigned &accesses, unsigned &misses, unsigned &depthColorWrites, unsigned &blendinghColorWrites) const;
//private:
//	std::string m_name;
//	const cache_config &m_config;
//	z_tag_array  m_tag_array;
//
//	//! interfacing with L2
//	mem_fetch_interface *m_memport;
//	mem_fetch_allocator *m_memfetch_creator;
//	//! miss queue for z-tag-array
//	std::list<mem_fetch*> m_miss_queue;
//	//! -ztag-array miss queue size
//	unsigned m_miss_queue_size;
//	//! buffer entry status in incoming request buffer
//	typedef enum {
//		INVALID,
//		VALID,
//		BLOCKED,
//		ISSUED
//	}request_buffer_entry_status;
//
//	//! Incoming request buffer entry
//	class request_buffer_entry{
//	public:
//		mem_fetch* mf;
//		new_addr_type back_up_C_address;
//		request_buffer_entry_status status;
//		unsigned index;
//		unsigned arriving_time;
//		request_buffer_entry(){
//			arriving_time = (unsigned)-1;
//			mf = NULL;
//			status = INVALID;
//		}
//	};
//
//	//! Input request buffer (In-fly requests)
//	class request_buffer{
//		unsigned time;
//		request_buffer_entry* m_request_buffer;
//		unsigned size;
//	public:
//		z_cache* m_z_cache;
//		request_buffer(unsigned size){
//			time = 0;
//			this->size = size;
//			m_request_buffer = new request_buffer_entry[size];
//
//		}
//		//! return value: is something scheduled request: scheduled entry
//		bool request_scheduler(request_buffer_entry*& request);
//		//! insert request into buffer
//		bool insert_request(mem_fetch* mf);
//		//! Release entry of buffer
//		void release_entry(request_buffer_entry* entry);
//		void release_next_waiting_entry(request_buffer_entry* entry);
//		bool full();
//		void dump() const;
//		friend class z_unit;
//		friend class z_cache;
//	}m_request_buffer;
//	//! Select one entry from request buffer and apply it into z-tag array
//	void issue_request(unsigned cycle);
//
//
//	//! buffer entry of waiting request fir z-test with ready data
//	class z_check_buffer_entry{
//	public:
//		mem_fetch* mf;
//		mem_fetch* color_generated_mf;
//		bool m_valid;
//		bool m_ready;
//		new_addr_type m_block_addr;
//		unsigned m_cache_index;
//		unsigned m_data_size;
//		request_buffer_entry* m_request_buffer_entry;
//		z_check_buffer_entry(){m_valid = false;color_generated_mf = NULL;}
//		z_check_buffer_entry( new_addr_type a, unsigned i, unsigned d )
//		{
//			set(a,i,d);
//			color_generated_mf = NULL;
//		}
//		z_check_buffer_entry(mem_fetch* mf, new_addr_type a, unsigned i, unsigned d )
//		{
//			set(mf,a,i,d);
//			color_generated_mf = NULL;
//		}
//		void set( new_addr_type a, unsigned i, unsigned d )
//		{
//			m_valid = true;
//			m_ready = false;
//			m_block_addr = a;
//			m_cache_index = i;
//			m_data_size = d;
//			mf = NULL;
//		}
//		void set(mem_fetch* mf, new_addr_type a, unsigned i, unsigned d )
//		{
//			m_valid = true;
//			m_ready = false;
//			m_block_addr = a;
//			m_cache_index = i;
//			m_data_size = d;
//			this->mf = mf;
//		}
//	};
//
//	//! z_check buffer
//	class z_check_buffer{
//	public:
//		void dump() const;
//		z_check_buffer(int size){
//			this->size = size;
//		}
//		bool full(){
//			return ((int)m_buffer.size()>= size);
//		}
//		bool empty(){
//			return m_buffer.empty();
//		}
//		int get_next_entry_number(){
//			int index = -1;
//			unsigned min = (unsigned)-1;
//			for(int i = 0; i < (int)m_buffer.size(); i++){
//				if(m_buffer[i].m_valid &&m_buffer[i].m_ready){
//					if(m_buffer[i].m_request_buffer_entry->arriving_time < min){
//						min = m_buffer[i].m_request_buffer_entry->arriving_time;
//						index = i;
//					}
//				}
//			}
//			return index;
//		}
//		z_check_buffer_entry* allocate(mem_fetch* mf, new_addr_type addr, unsigned index, unsigned data_size ){
//			assert(!full());
//			z_check_buffer_entry new_entry(mf,addr,index,data_size);
//			m_buffer.push_back(new_entry);
//			return &(m_buffer[m_buffer.size() -1]);
//		}
//		void release_entry(z_check_buffer_entry* d){
//			int index = -1;
//			for(int i = 0; i < (int)m_buffer.size(); i++){
//				if(m_buffer[i].mf == d->mf){
//					index = i;
//					break;
//				}
//			}
//			assert(index != -1);
//			m_buffer.erase(m_buffer.begin()+index);
//		}
//		void release_entry(mem_fetch* d){
//			int index = -1;
//			for(int i = 0; i < (int)m_buffer.size(); i++){
//				if(m_buffer[i].mf == d){
//					index = i;
//					break;
//				}
//			}
//			assert(index != -1);
//			m_buffer.erase(m_buffer.begin()+index);
//		}
//		z_check_buffer_entry* get_entry(int index){
//			assert(index < (int) m_buffer.size());
//			return &(m_buffer[index]);
//		}
//		void release_entry(int index){
//			assert(index < (int) m_buffer.size());
//			m_buffer.erase(m_buffer.begin()+index);
//		}
//		int get_entry_number(mem_fetch* mf){
//			for(int i = 0; i < (int)m_buffer.size(); i++){
//				if(m_buffer[i].m_valid && m_buffer[i].mf == mf)
//					return i;
//			}
//			assert(false);
//			return 0;
//		}
//		int get_entry_number_for_C_request(mem_fetch* mf){
//			for(int i = 0; i < (int)m_buffer.size(); i++){
//				if(m_buffer[i].m_valid && m_buffer[i].color_generated_mf == mf)
//					return i;
//			}
//			assert(false);
//			return 0;
//		}
//		bool contain(mem_fetch* mf){
//			for(int i = 0; i < (int)m_buffer.size(); i++){
//				if(m_buffer[i].m_valid && m_buffer[i].mf == mf)
//					return true;
//			}
//			return false;
//		}
//		void mark_ready(int index){
//			m_buffer[index].m_ready = true;
//		}
//		z_check_buffer_entry* element(int index){
//			assert(index >=0 && index < size);
//			return &(m_buffer[index]);
//		}
//	private:
//		std::vector<z_check_buffer_entry> m_buffer;
//		int size;
//		//! Z-value ready to check with z-checker unit
//
//	};
//	//! buffer which holds ready z-values to compare with requests and update
//	z_check_buffer m_z_check_buffer;
//
//	//! check Z value and update Z and C value if required
//	void z_checker(unsigned cycle);
//	//! insert miss requests to the L2 buffer
//	void miss_queue_cycle();
//	const memory_config* m_memory_config;
//public:
//	void mark_z_access_ready(mem_fetch* mf);
//	void release_request_entry(mem_fetch* mf);
//        unsigned m_color_writes_depth_test;
//        unsigned m_color_writes_blending;
//	friend class z_unit;
//};
//
//
//class z_unit{
//public:
//	z_unit(const char* name, const memory_config * mem_config,mem_fetch_interface* mf_interface,
//			mem_fetch_allocator* mf_allocator);
//	bool insert_into_incoming_buffer(mem_fetch* mf){
//		return m_z_cache->insert_into_incoming_buffer(mf);
//	}
//	void cycle(unsigned cycle){
//		m_z_cache->cycle(cycle);
//	}
//	void mark_z_access_ready(mem_fetch* mf){
//		m_z_cache->mark_z_access_ready(mf);
//	}
//	void release_request_entry(mem_fetch* mf){
//		m_z_cache->release_request_entry(mf);
//	}
//	bool incoming_buffer_full(){
//		return m_z_cache->m_request_buffer.full();
//	}
//        void print(FILE *fp, unsigned &accesses, unsigned &misses, unsigned &depthColorWrites, unsigned &blendingColorWrites) const
//        {
//           m_z_cache->print(fp,accesses,misses,depthColorWrites,blendingColorWrites);
//        }
//private:
//	//const z_unit_config& m_z_unit_config;
//	const memory_config* m_memory_config;
//	z_cache* m_z_cache;
//};
//
//class z_unit_mf_allocator : public mem_fetch_allocator {
//public:
//	z_unit_mf_allocator( const memory_config *config )
//	{
//		m_memory_config = config;
//	}
//	virtual mem_fetch * alloc(const class warp_inst_t &inst, const mem_access_t &access) const
//	{
//		abort();
//		return NULL;
//	}
//	virtual mem_fetch * alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr) const{
//		mem_access_t access( type, addr, size, wr );
//		mem_fetch *mf = new mem_fetch( access,
//				NULL,
//				WRITE_PACKET_SIZE,
//				-1,
//				-1,
//				-1,
//				m_memory_config );
//		return mf;
//	}
//private:
//	const memory_config *m_memory_config;
//};
//#endif
