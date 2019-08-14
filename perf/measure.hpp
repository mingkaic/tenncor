#include <chrono>
#include <string>
#include <unordered_map>
#include <ostream>

#include "jobs/scope_guard.hpp"

#ifndef PERF_MEASURE_HPP
#define PERF_MEASURE_HPP

namespace perf
{

using TimeT = std::chrono::high_resolution_clock::time_point;

using DurationT = std::chrono::duration<long,std::milli>;

using MeanDurT = std::pair<DurationT,size_t>;

struct PerfRecord final
{
	void to_csv (std::ostream& out) const
	{
		for (const auto& durs : durations_)
		{
			const auto& meandur = durs.second;
			out << durs.first << "," << (meandur.first / meandur.second).count() << "\n";
		}
	}

	void record_duration (std::string fname, DurationT duration)
	{
		auto it = durations_.find(fname);
		if (durations_.end() == it)
		{
			durations_.emplace(fname, MeanDurT{duration, 1});
		}
		else
		{
			auto& meandur = it->second;
			meandur.first += duration;
			++meandur.second;
		}
	}

	bool empty (void) const
	{
		return durations_.empty();
	}

private:
	std::unordered_map<std::string,MeanDurT> durations_;
};

struct MeasureScope final : public jobs::ScopeGuard
{
	MeasureScope (PerfRecord* record, std::string fname) :
		jobs::ScopeGuard([this, fname]()
		{
			this->record_->record_duration(fname,
				std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::high_resolution_clock::now() -
					this->measure_start_));
		}),
		record_(record),
		measure_start_(std::chrono::high_resolution_clock::now()) {}

private:
	PerfRecord* record_;

	TimeT measure_start_;
};

PerfRecord& get_global_record (void);

#define MEASURE(NAME)perf::MeasureScope _defer(&perf::get_global_record(), NAME);

}

#endif // PERF_MEASURE_HPP
