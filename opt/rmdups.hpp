#include "opt/stats.hpp"

#ifndef OPT_RMDUPS_HPP
#define OPT_RMDUPS_HPP

namespace opt
{

void replace_parents (const teq::ParentFinder& pfinder,
	teq::iTensor* source, teq::TensptrT target);

template <typename T>
std::vector<T> remove_duplicates (teq::TensptrsT& roots, std::vector<T> tens,
	const teq::ParentFinder& pfinder,
	tag::TagRegistry& registry = tag::get_reg())
{
	if (tens.empty())
	{
		return {};
	}

	teq::TensSetT priorities;
	std::unordered_map<teq::iTensor*,std::vector<size_t>> rindices;
	for (size_t i = 0, n = roots.size(); i < n; ++i)
	{
		teq::TensptrT& root = roots[i];
		priorities.emplace(root.get());
		rindices[root.get()].push_back(i);
	}

	std::sort(tens.begin(), tens.end(),
		[&priorities](T& a, T& b) { return lt(priorities, a.get(), b.get()); });
	T last = tens[0];
	std::vector<T> uniques = {last};
	size_t n = tens.size();
	uniques.reserve(n - 1);
	for (size_t i = 1; i < n; ++i)
	{
		T& cur = tens[i];
		if (is_equal(last.get(), cur.get()))
		{
			logs::debugf("replacing %s", cur->to_string().c_str());
			// remove equivalent node
			replace_parents(pfinder, cur.get(), last);

			auto it = rindices.find(cur.get());
			if (rindices.end() != it)
			{
				for (size_t ridx : it->second)
				{
					roots[ridx] = last;
				}
			}

			// todo: mark parents as uninitialized, reinitialize entire graph, or uninitialize everything to begin with

			// inherit tags
			registry.move_tags(last, cur.get());
		}
		else
		{
			uniques.push_back(cur);
			last = cur;
		}
	}
	return uniques;
}

using ImmutablesT = std::vector<teq::LeafptrT>;

using HFunctorsT = std::vector<std::vector<teq::FuncptrT>>;

// identify immutable leaves and organize functors by maxheight
void populate_graph (ImmutablesT& immutables, HFunctorsT& functors,
	const teq::TensptrsT& roots);

// delete and update equivalent immutable leaves and functors
void remove_all_duplicates (teq::TensptrsT& roots,
	ImmutablesT& immutables, HFunctorsT& functors);

}

#endif // OPT_RMDUPS_HPP
