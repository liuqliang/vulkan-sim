#include "rt_func_units.h"
#include "../../libcuda/gpgpu_context.h"


std::string RTFuncInsnTypeNames[] = {
    "RT_DECODE",
    "RT_VEC_SUB",
    "RT_RCP",
    "RT_MUL",
    "RT_MAXMIN",
    "RT_MINMAX",
    "RT_VEC_CMP",
    "RT_VEC_OR",
    "RT_CROSS",
    "RT_DOT",
    "RT_SQRT",
    "RT_RAY_BOX_FUNC_OP",
    "RT_RAY_TRI_FUNC_OP",
    "RT_RAY_XFORM_FUNC_OP"
};

rt_op_unit::rt_op_unit(unsigned sid, unsigned unit_id, unsigned latency, unsigned init_cycles, RTFuncInsnType op, std::string unit_name, unsigned n_units, op_unit_stats* stats) {
    m_sid = sid;
    m_unit_id = unit_id;
    m_unit_name = unit_name;
    m_n_units = n_units;
    m_latency = latency;
    m_init_cycles = init_cycles;
    m_op = op;
    m_waiting_cycles = (unsigned *)calloc(n_units, sizeof(unsigned));
    m_stats = stats;

    printf("[%d:%d] Created %d %s unit with latency %d and init_cycles %d\n", m_sid, m_unit_id, m_n_units, m_unit_name.c_str(), m_latency, m_init_cycles);
}

void rt_op_unit::cycle() {
    for (unsigned i=0; i<m_n_units; i++) {
        if (m_waiting_cycles[i] > 0) {
            m_waiting_cycles[i]--;
            if (m_input_queue.size() > 0) {
                m_stats->pipeline_stalled_cycles++;
            }
        }
        else {
            if (m_input_queue.size() > 0) {
                warp_thread_id next_thread = m_input_queue.front();
                m_current_execution.push_back(std::pair<warp_thread_id, unsigned>(next_thread, m_latency));
                m_input_queue.pop();
                printf("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_unit_name.c_str());

                if (m_input_queue.size() > 0) {
                    m_stats->input_stalled_cycles++;
                }
            }
        }
    }

    if (m_current_execution.size() > 0) {
        m_stats->total_active_cycles++;
    }

    // Cycle pipeline
    for (auto it=m_current_execution.begin(); it!=m_current_execution.end();) {
        if (it->second > 0) {
            it->second--;
            it++;
        }
        else {
            m_output_queue.push(it->first);
            it = m_current_execution.erase(it);
            printf("[%d:%d] finished in %s.\n", it->first.warp_uid, it->first.tid, m_unit_name.c_str());
            m_stats->total_ops++;
        }
    }
}

rt_func_unit::rt_func_unit(unsigned sid, rt_func_unit_config config, func_unit_stats* stats, unsigned unit_id) {
    m_sid = sid;
    m_unit_id = unit_id;
    m_config = config;
    m_active_threads = 0;
    m_stats = stats;

    m_op_stats = m_stats->init_per_op(m_unit_id, static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE));

    // Iterate through each op type
    printf("Creating RT op units...\n");
    unsigned op_unit_id = 0;
    for (unsigned i=0; i<static_cast<unsigned>(RTFuncInsnType::RT_MAX_INSN_TYPE); i++) {
        unsigned n_units = m_config.n_units[i];
        unsigned latency = m_config.latency[i];
        unsigned init_cycles = m_config.initiation_cycles[i];

        rt_op_unit* new_unit = new rt_op_unit(m_sid, op_unit_id, latency, init_cycles, static_cast<RTFuncInsnType>(i), RTFuncInsnTypeNames[i], n_units, &m_op_stats[i]);
        m_op_units.push_back(new_unit);
        op_unit_id++;
    }
}

void rt_func_unit::cycle(std::map<unsigned, warp_inst_t>& warps, std::deque<warp_thread_id>& awaiting_threads, std::deque<std::pair<unsigned, new_addr_type> > &store_queue) {

    // Process one thread (for now)
    if (awaiting_threads.size() > 0) {
        warp_thread_id next_thread = awaiting_threads.front();

        // Make an uid (tid is 32 max, which is 5 bits)
        unsigned next_thread_uid = next_thread.warp_uid << 5 | next_thread.tid;
        unsigned long long current_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
        latency_tracker[next_thread_uid] = current_cycle;

        m_active_threads++;
        awaiting_threads.pop_front();

        RTFuncInsnType next_op = warps[next_thread.warp_uid].get_next_op(next_thread.tid);
        m_op_units[static_cast<unsigned>(next_op)]->add_input(next_thread);

        RT_DPRINTF("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_op_units[static_cast<unsigned>(next_op)]->get_unit_name().c_str());
    }

    // Cycle each op unit
    for (auto it=m_op_units.begin(); it!=m_op_units.end(); it++) {
        (*it)->cycle();
        if ((*it)->has_output()) {
            warp_thread_id next_thread = (*it)->get_output();
            bool done = warps[next_thread.warp_uid].dec_thread_latency(next_thread.tid);

            if (done) {
                // Get stores
                std::deque<std::pair<unsigned, new_addr_type> > st_queue = warps[next_thread.warp_uid].check_stores(next_thread.tid);
                std::copy(st_queue.begin(), st_queue.end(), std::back_inserter(store_queue));

                // Remove from func_unit
                m_active_threads--;
                m_stats->total_intersections[m_unit_id]++;

                // Remove from latency tracker
                unsigned next_thread_uid = next_thread.warp_uid << 5 | next_thread.tid;
                unsigned long long current_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
                unsigned long long latency = current_cycle - latency_tracker[next_thread_uid];
                latency_tracker.erase(next_thread_uid);

                TransactionType intersection_type = warps[next_thread.warp_uid].get_current_transaction_type(next_thread.tid);
                m_stats->intersection_cycles[static_cast<unsigned>(intersection_type)] += latency;
                m_stats->intersection_count[static_cast<unsigned>(intersection_type)]++;

                printf("[%d:%d] finished\n", next_thread.warp_uid, next_thread.tid);
            }
            else {
                RTFuncInsnType next_op = warps[next_thread.warp_uid].get_next_op(next_thread.tid);
                m_op_units[static_cast<unsigned>(next_op)]->add_input(next_thread);
                printf("Adding [%d:%d] to %s.\n", next_thread.warp_uid, next_thread.tid, m_op_units[static_cast<unsigned>(next_op)]->get_unit_name().c_str());
            }
        }
    }
}

void func_unit_stats::print(FILE *fout) {
    unsigned intersections = 0;
    fprintf(fout, "Per func unit intersections: ");
    for (unsigned i=0; i<m_n_units; i++) {
        fprintf(fout, "%d ", total_intersections[i]);
        intersections += total_intersections[i];
    }
    fprintf(fout, "\n");
    fprintf(fout, "total_intersections = %d\n", intersections);

    for (unsigned t=0; t<static_cast<unsigned>(TransactionType::UNDEFINED); t++) {
        float avg_latency = (float)intersection_cycles[t] / (float)intersection_count[t];
        if (!std::isnan(avg_latency)) {
            fprintf(fout, "avg_latency[%d] = %f\n", t, avg_latency);
        }
    }

    for (unsigned i=0; i<m_n_op_units; i++) {
        fprintf(fout, "total_active_cycles[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%llu ", per_op_stats[j][i].total_active_cycles);
        }
        fprintf(fout, "\n");
        fprintf(fout, "pipeline_stalled_cycles[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%llu ", per_op_stats[j][i].pipeline_stalled_cycles);
        }
        fprintf(fout, "\n");
        fprintf(fout, "input_stalled_cycles[%s] = ", RTFuncInsnTypeNames[i].c_str());
        for (unsigned j=0; j<m_n_units; j++) {
            fprintf(fout, "%llu ", per_op_stats[j][i].input_stalled_cycles);
        }
        fprintf(fout, "\n");
    }

}
