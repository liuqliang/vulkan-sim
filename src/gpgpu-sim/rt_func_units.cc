#include "rt_func_units.h"
#include "../../libcuda/gpgpu_context.h"

rt_func_unit::rt_func_unit(unsigned sid, rt_func_unit_config config) {
    m_sid = sid;
    m_config = config;
    m_waiting_cycles = 0;
}

void rt_func_unit::cycle(std::map<unsigned, warp_inst_t>& warps, std::deque<warp_thread_id>& awaiting_threads, std::deque<std::pair<unsigned, new_addr_type> > &store_queue) {
    // First add awaiting threads if ready
    if (m_waiting_cycles == 0) {
        if (awaiting_threads.size() > 0) {
            warp_thread_id next_thread = awaiting_threads.front();
            m_current_execution.push_back(next_thread);
            awaiting_threads.pop_front();
            m_waiting_cycles = m_config.initiation_cycles;
            RT_DPRINTF("Adding [%d:%d] to m_current_execution.\n", next_thread.warp_uid, next_thread.tid);
        }
    }
    else {
        assert(m_waiting_cycles > 0); // Should always be positive
        m_waiting_cycles--;
    }

    // Cycle pipeline
    for (auto it=m_current_execution.begin(); it!=m_current_execution.end();) {
        warp_thread_id thread = *it;
        bool done = warps[thread.warp_uid].dec_thread_latency(thread.tid);
        RT_DPRINTF("Processing [%d:%d] (done? %d)\n", thread.warp_uid, thread.tid, done);

        if (done) {
            // Get stores
            std::deque<std::pair<unsigned, new_addr_type> > st_queue = warps[thread.warp_uid].check_stores(thread.tid);
            std::copy(st_queue.begin(), st_queue.end(), std::back_inserter(store_queue));

            // Remove from func_unit
            it = m_current_execution.erase(it);
            RT_DPRINTF("Removing...\n");
        }
        else {
            it++;
        }
    }
}


