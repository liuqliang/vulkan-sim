#ifndef RT_FUNC_UNITS_INCLUDED
#define RT_FUNC_UNITS_INCLUDED

#include "../abstract_hardware_model.h"


typedef struct warp_thread_id {
    unsigned warp_uid;
    unsigned tid;
} warp_thread_id;

typedef struct rt_func_unit_config {
    TransactionType type;
    unsigned initiation_cycles;
} rt_func_unit_config;

class rt_func_unit {
    public:
        rt_func_unit(unsigned sid, rt_func_unit_config config);
        ~rt_func_unit();

        void cycle(std::map<unsigned, warp_inst_t>& warps, std::deque<warp_thread_id>& awaiting_threads, std::deque<std::pair<unsigned, new_addr_type> > &store_queue);
        int get_type() { return (int)m_config.type; }
        unsigned count_in_progress() { return m_current_execution.size(); }

    private:
        rt_func_unit_config m_config;
        std::vector<warp_thread_id> m_current_execution;
        unsigned m_waiting_cycles;

        // administrative details
        unsigned m_sid;
};

#endif
