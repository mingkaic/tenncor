
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

using TxConsumeF = std::function<void(StatsMapT&,const StatsMapT&)>;

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

inline void bind_stats (StatsMapT& out,
	const teq::TensSetT& candidates, const Stats& stats)
{
	std::transform(candidates.begin(), candidates.end(),
		std::inserter(out, out.end()),
		[&](teq::iTensor* tens)
		{
			return std::pair<teq::iTensor*,Stats>{tens, stats};
		});
}

inline void union_stats (StatsMapT& out, const StatsMapT& other)
{
	for (auto& opair : other)
	{
		if (estd::has(out, opair.first))
		{
			out[opair.first].merge(opair.second);
		}
		else
		{
			out.emplace(opair);
		}
	}
}

inline void intersect_stats (StatsMapT& out, const StatsMapT& other)
{
	for (auto it = out.begin(), et = out.end(); it != et;)
	{
		if (estd::has(other, it->first))
		{
			it->second.merge(other.at(it->first));
			++it;
		}
		else
		{
			it = out.erase(it);
		}
	}
}

}

#endif // QUERY_STATS_HPP
