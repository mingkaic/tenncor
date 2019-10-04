///
/// rmdups.hpp
/// opt
///
/// Purpose:
/// Define TEQ functor duplication removal algorithm
///

#include "opt/stats.hpp"

#ifndef OPT_RMDUPS_HPP
#define OPT_RMDUPS_HPP

namespace opt
{

/// Replace source tensor's position with target's position
/// in the sense that all parents of source (found in pfinder)
/// take on target as the new child in place of source's
void replace_parents (const teq::ParentFinder& pfinder,
	teq::iTensor* source, teq::TensptrT target);

/// Return non-duplicate nodes of a HeightMatrix row (tens)
/// If T is a functor, a functor X is duplicate if their any of their child
/// have another parent of the same opcode as X that has identical arguments
/// as X (order matters if X is non-commutative)
/// If T is a leaf, a leaf X is duplicate if the leaf is a constant and there
/// exists another constant that has the same shape and data
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

/// Vector of presumably immutable leaves
using ImmutablesT = std::vector<teq::LeafptrT>;

/// Matrix of functors
using HFunctorsT = std::vector<std::vector<teq::FuncptrT>>;

/// Populate immutables with immutable leaves and functors with functors
/// ordered by functor max height in ascending order
void populate_graph (ImmutablesT& immutables, HFunctorsT& functors,
	const teq::TensptrsT& roots);

/// Delete and update equivalent immutable leaves and functors
/// according to remove_duplicates
void remove_all_duplicates (teq::TensptrsT& roots,
	ImmutablesT& immutables, HFunctorsT& functors);

}

#endif // OPT_RMDUPS_HPP
