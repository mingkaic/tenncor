#include <queue>

#include "ade/iopfunc.hpp"

#include "experimental/cce/partition.hpp"

#ifdef CCE_PARTITION_HPP

namespace cce
{

struct WeighedGroup final
{
	std::vector<ade::iFunctor*> reps_;

	std::unordered_set<ade::iTensor*> ancestors_;

	double weight_;
};

PartGroupsT k_partition (ade::TensT roots, size_t k, OpWeightT weights)
{
	PartGroupsT groups;

	ade::GraphStat stat;
	ade::ParentFinder pfinder;
	for (auto root : roots)
	{
		root->accept(stat);
		root->accept(pfinder);
	}

	// partition by bases (the funcs right above variables)
	std::vector<ade::iFunctor*> bases;
	for (auto& gpair : stat.graphsize_)
	{
		if (gpair.second.upper_ == 1)
		{
			bases.push_back(static_cast<ade::iFunctor*>(gpair.first));
		}
	}

	// partition bases by number of ancestor
	std::unordered_map<ade::iTensor*,double> weight_map;
	std::unordered_map<ade::iFunctor*,
		std::unordered_set<ade::iTensor*>> ancestors;
	for (auto base : bases)
	{
		std::queue<ade::iTensor*> q;
		q.push(base);
		while (false == q.empty())
		{
			auto tens = q.front();
			if (false == estd::has(weight_map, tens))
			{
				double weight = 1;
				if (auto op = dynamic_cast<ade::iOperableFunc*>(tens))
				{
					weight = estd::try_get(weights, op->type_code(), 1);
				}
				weight_map.emplace(tens, weight);
			}
			ade::ParentFinder::ParentMapT parents;
			if (estd::get(parents, pfinder.parents_, tens))
			{
				for (auto& ppair : parents)
				{
					ancestors[base].emplace(ppair.first);
					q.push(ppair.first);
				}
			}
			q.pop();
		}
	}

	size_t nbases = bases.size();
	if (nbases <= k)
	{
		groups = PartGroupsT(nbases);
		for (size_t i = 0; i < nbases; ++i)
		{
			auto& ancs = ancestors[bases[i]];

			auto& group = groups[i];
			group.reserve(ancs.size() + 1);
			group.push_back(bases[i]);
			for (auto anc : ancs)
			{
				group.push_back(static_cast<ade::iFunctor*>(anc));
			}
		}
	}
	else
	{
		// pick top K groups by minimizing sum weight of unique ancestors
		std::vector<WeighedGroup> base_groups;
		base_groups.reserve(nbases);
		for (auto& base : bases)
		{
			auto& ancs = ancestors.at(base);
			double bweight = weight_map.at(base);
			for (auto& anc : ancs)
			{
				bweight += weight_map.at(anc);
			}
			base_groups.push_back(WeighedGroup{{base}, ancs, bweight});
		}
		auto group_cmp =
			[](const WeighedGroup& a, const WeighedGroup& b)
			{
				return a.weight_ < b.weight_;
			};
		auto group_mincmp =
			[](const WeighedGroup& a, const WeighedGroup& b)
			{
				return a.weight_ > b.weight_;
			};

		std::make_heap(base_groups.begin(), base_groups.end(), group_cmp);
		std::vector<WeighedGroup> kgroups;
		for (size_t i = 0; i < k; ++i)
		{
			kgroups.push_back(base_groups.front());
			std::push_heap(kgroups.begin(), kgroups.end(), group_mincmp);

			std::pop_heap(base_groups.begin(), base_groups.end(), group_cmp);
			base_groups.pop_back();
		}
		for (size_t i = k; i < nbases; ++i)
		{
			auto smallest_kgroup = kgroups.front();
			auto& smallest_bgroup = base_groups.front();
			// join the smallest groups
			smallest_kgroup.weight_ += weight_map.at(smallest_bgroup.reps_[0]);
			auto& bg_ancs = smallest_bgroup.ancestors_;
			for (auto& bg_anc : bg_ancs)
			{
				if (false == estd::has(smallest_kgroup.ancestors_, bg_anc))
				{
					smallest_kgroup.weight_ += weight_map.at(bg_anc);
					smallest_kgroup.ancestors_.emplace(bg_anc);
				}
			}
			smallest_kgroup.reps_.push_back(smallest_bgroup.reps_[0]);

			std::pop_heap(kgroups.begin(), kgroups.end(), group_mincmp);
			kgroups[k - 1] = smallest_kgroup;
			std::push_heap(kgroups.begin(), kgroups.end(), group_mincmp);

			std::pop_heap(base_groups.begin(), base_groups.end(), group_cmp);
			base_groups.pop_back();
		}
		groups.reserve(nbases);
		for (auto& kgroup : kgroups)
		{
			std::vector<ade::iFunctor*> group;
			group.reserve(kgroup.reps_.size() + kgroup.ancestors_.size());
			for (auto& rep : kgroup.reps_)
			{
				group.push_back(rep);
			}
			for (auto& anc : kgroup.ancestors_)
			{
				group.push_back(static_cast<ade::iFunctor*>(anc));
			}
			groups.push_back(group);
		}
	}

	// for each group, order by upper height
	for (auto& group : groups)
	{
		std::sort(group.begin(), group.end(),
			[&stat](ade::iTensor* a, ade::iTensor* b)
			{
				return stat.graphsize_[a].upper_ < stat.graphsize_[b].upper_;
			});
	}

	return groups;
}

}

#endif
