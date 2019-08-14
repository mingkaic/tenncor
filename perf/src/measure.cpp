#include "perf/measure.hpp"

#ifdef PERF_MEASURE_HPP

namespace perf
{

PerfRecord& get_global_record (void)
{
	static PerfRecord record;
	return record;
}

}

#endif
