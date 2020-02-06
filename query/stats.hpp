
#ifndef QUERY_STATS_HPP
#define QUERY_STATS_HPP

namespace query
{

struct Stats final
{
	friend bool operator == (const Stats& a, const Stats& b)
	{
		return a.depth_ == b.depth_ &&
			a.path_total_ == b.path_total_ &&
			a.base_str() == b.base_str();
	}

	friend bool operator < (const Stats& a, const Stats& b)
	{
		if (a.depth_ == b.depth_)
		{
			if (a.path_total_ == b.path_total_)
			{
				return a.base_str() < b.base_str();
			}
			return a.path_total_ < b.path_total_;
		}
		return a.depth_ < b.depth_;
	}

	friend bool operator > (const Stats& a, const Stats& b)
	{
		if (a.depth_ == b.depth_)
		{
			if (a.path_total_ == b.path_total_)
			{
				return a.base_str() > b.base_str();
			}
			return a.path_total_ > b.path_total_;
		}
		return a.depth_ > b.depth_;
	}

	void merge (const Stats& other)
	{
		depth_ = std::max(depth_, other.depth_);
		path_total_ += other.path_total_;
		bases_.insert(bases_.end(),
			other.bases_.begin(), other.bases_.end());
	}

	std::string base_str (void) const
	{
		std::vector<std::string> reps;
		reps.reserve(bases_.size());
		std::transform(bases_.begin(), bases_.end(),
			std::back_inserter(reps),
			[](const teq::iTensor* base)
			{ return base->to_string() + base->shape().to_string(); });
		return fmts::to_string(reps.begin(), reps.end());
	}

	size_t depth_;

	size_t path_total_;

	teq::CTensT bases_;
};

using StatsMapT = teq::TensMapT<Stats>;

using TxConsumeF = std::function<void(StatsMapT&,const StatsMapT&)>;

inline void bind_stats (StatsMapT& out, const teq::TensSetT& candidates,
	const teq::iTensor* base, size_t pathlen)
{
	std::transform(candidates.begin(), candidates.end(),
		std::inserter(out, out.end()),
		[&](teq::iTensor* root)
		{
			return std::pair<teq::iTensor*,Stats>{root,
				Stats{pathlen, pathlen, teq::CTensT{base}}};
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
