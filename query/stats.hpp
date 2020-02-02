
#ifndef QUERY_STATS_HPP
#define QUERY_STATS_HPP

namespace query
{

struct Stats final
{
	friend bool operator == (const Stats& a, const Stats& b)
	{
		return a.depth_ == b.depth_ && a.path_total_ == b.path_total_;
	}

	friend bool operator < (const Stats& a, const Stats& b)
	{
		if (a.depth_ == b.depth_)
		{
			return a.path_total_ < b.path_total_;
		}
		return a.depth_ < b.depth_;
	}

	friend bool operator > (const Stats& a, const Stats& b)
	{
		if (a.depth_ == b.depth_)
		{
			return a.path_total_ > b.path_total_;
		}
		return a.depth_ > b.depth_;
	}

	void merge (const Stats& other)
	{
		depth_ = std::max(depth_, other.depth_);
		path_total_ += other.path_total_;
	}

	size_t depth_;

	size_t path_total_;
};

using StatsMapT = teq::TensMapT<Stats>;

inline void merge_stats (StatsMapT& out,
	const teq::TensSetT& candidates, const Stats& stats)
{
	for (teq::iTensor* cand : candidates)
	{
		if (estd::has(out, cand))
		{
			out[cand].merge(stats);
		}
		else
		{
			out.emplace(cand, stats);
		}
	}
}

}

#endif // QUERY_STATS_HPP
