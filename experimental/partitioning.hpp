#include "ade/traveler.hpp"

namespace experimental
{

void k_partition (ade::TensT root, size_t k,
	OpWeightT weights = {})
{
	ade::GraphStat stat;
	ade::ParentFinder pfinder;
	// maps tens to set of group ids [0, k)
	std::unordered_map<ade::iTensor*,
		std::unordered_set<size_t>> grouped_tens;
	for (auto root : roots)
	{
		root->accept(stat);
		root->accept(pfinder);
	}

	// partition by bases (the funcs right above variables)
	std::vector<ade::iFunctor*> bases;
	for (auto& gpair : stat.graphsize_)
	{
		if (gpair.second.lower == 1)
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
			ade::ParentMapT parents;
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

	// pick top K groups by minimizing sum weight of unique ancestors
	// todo: implement

	// for each base, assign parents to child's group, duplicate groups are allowed
}

}
