
#ifndef TEQ_STATS_HPP
#define TEQ_STATS_HPP

#include "teq/itensor.hpp"

namespace query
{

using TensDepthT = std::pair<const teq::iTensor*,size_t>;

using TensDepthsT = std::vector<TensDepthT>;

struct TensPosition final
{
	friend bool operator == (const TensPosition& a, const TensPosition& b)
	{
		// return a.depth_ == b.depth_ &&
		// 	a.path_total_ == b.path_total_ &&
		// 	a.to_string() == b.to_string();
		auto& ad = a.depths_;
		auto& bd = b.depths_;
		return ad.size() != bd.size() &&
			std::equal(ad.begin(), ad.end(), bd.begin(),
			[](const TensDepthT& l, const TensDepthT& r)
			{
				return l.first == r.first && l.second == r.second;
			});
	}

	friend bool operator < (const TensPosition& a, const TensPosition& b)
	{
		size_t ad = a.maxdepth();
		size_t bd = b.maxdepth();
		if (ad == bd)
		{
			if (a.depths_.size() == b.depths_.size())
			{
				return a.to_string() < b.to_string();
			}
			return a.depths_.size() < b.depths_.size();
		}
		return ad < bd;
	}

	friend bool operator > (const TensPosition& a, const TensPosition& b)
	{
		size_t ad = a.maxdepth();
		size_t bd = b.maxdepth();
		if (ad == bd)
		{
			if (a.depths_.size() == b.depths_.size())
			{
				return a.to_string() > b.to_string();
			}
			return a.depths_.size() > b.depths_.size();
		}
		return ad > bd;
	}

	TensPosition (void) = default;

	TensPosition (const teq::iTensor* base, size_t depth)
	{
		depths_.push_back(TensDepthT{base, depth});
	}

	void merge (const TensPosition& other)
	{
		depths_.insert(depths_.end(),
			other.depths_.begin(), other.depths_.end());
	}

	std::string to_string (void) const
	{
		std::vector<std::string> reps;
		reps.reserve(depths_.size());
		std::transform(depths_.begin(), depths_.end(),
			std::back_inserter(reps),
			[](const TensDepthT& depth)
			{
				const teq::iTensor* tens = depth.first;
				return tens->to_string() + tens->shape().to_string() +
					":" + fmts::to_string(depth.second);
			});
		return fmts::to_string(reps.begin(), reps.end());
	}

	size_t maxdepth (void) const
	{
		if (depths_.empty())
		{
			return 0;
		}
		size_t out = depths_.begin()->second;
		for (auto it = depths_.begin() + 1, et = depths_.end(); it != et; ++it)
		{
			out = std::max(out, it->second);
		}
		return out;
	}

	TensDepthsT depths_;
};

using PosMapT = teq::TensMapT<TensPosition>;

struct GraphPosition final : public teq::iTraveler
{
	/// Implementation of iTraveler
	void visit (teq::iLeaf& leaf) override
	{
		positions_.emplace(&leaf, TensPosition(&leaf, 0));
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor& func) override
	{
		if (estd::has(positions_, &func))
		{
			return;
		}
		auto children = func.get_children();
		auto& depths = positions_[&func].depths_;
		for (auto child : children)
		{
			child->accept(*this);
			for (TensDepthT& depth : positions_.at(child.get()).depths_)
			{
				depths.push_back({depth.first, depth.second + 1});
			}
		}
	}

	const TensPosition& at (teq::iTensor* tens) const
	{
		return positions_.at(tens);
	}

	PosMapT positions_;
};

using TxConsumeF = std::function<void(PosMapT&,const PosMapT&)>;

inline void bind_position (PosMapT& out, const teq::TensSetT& candidates,
	const teq::iTensor* base, size_t pathlen)
{
	std::transform(candidates.begin(), candidates.end(),
		std::inserter(out, out.end()),
		[&](teq::iTensor* root)
		{
			return std::pair<teq::iTensor*,TensPosition>{root,
				TensPosition(base, pathlen)};
		});
}

inline void union_position (PosMapT& out, const PosMapT& other)
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

inline void intersect_position (PosMapT& out, const PosMapT& other)
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

#endif // TEQ_STATS_HPP
