#include "opt/matcher.hpp"

#ifdef OPT_MATCHER_HPP

namespace opt
{

/// Return true if successfully merged a and b into out without conflicting values
static bool merge_woconflict (Candidate& out, const Candidate& a, const Candidate& b)
{
	for (auto apair : a.anys_)
	{
		teq::TensptrT bany;
		if (estd::get(bany, b.anys_, apair.first) &&
			bany != apair.second)
		{
			// found a conflicting value
			return false;
		}
		out.anys_.emplace(apair);
	}
	for (auto bpair : b.anys_)
	{
		if (false == estd::has(a.anys_, bpair.first))
		{
			out.anys_.emplace(bpair);
		}
	}
	return true;
}

void merge_cands (CandsT& out, const CandsT& a, const CandsT& b)
{
	CandsT next_cands;
	if (a.size() > 0 && b.size() > 0)
	{
		next_cands.reserve(a.size());
		// todo: make this efficient
		for (auto& cand1 : a)
		{
			for (auto& cand2 : b)
			{
				// check that ecands-cands intersection do not conflict
				Candidate next_cand;
				if (merge_woconflict(next_cand, cand1, cand2))
				{
					next_cands.push_back(next_cand);
				}
			}
		}
	}
	out = next_cands;
}

}

#endif
