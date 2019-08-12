#include <list>
#include <chrono>
#include <string>
#include <unordered_map>

#include "jobs/scope_guard.hpp"

#ifndef PERF_MEASURE_HPP
#define PERF_MEASURE_HPP

namespace perf
{

using TimeT = std::chrono::high_resolution_clock::time_point;

using DurationT = std::chrono::duration<long,std::milli>;

struct PerfRecord final
{
    void to_csv (std::ostream& out)
    {
        for (auto& durs : durations_)
        {
            out << durs.first << ",";
            double avg = 0;
            double n = durs.second.size();
            for (DurationT dur : durs.second)
            {
                avg += dur.count() / (double) n;
            }
            out << avg << "\n";
        }
    }

    void record_duration (std::string fname, DurationT duration)
    {
        durations_[fname].push_back(duration);
    }

private:
    std::unordered_map<std::string,std::list<DurationT>> durations_;
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

static PerfRecord global_record;

#define MEASURE(NAME)perf::MeasureScope _defer(&perf::global_record, NAME);

}

#endif // PERF_MEASURE_HPP
