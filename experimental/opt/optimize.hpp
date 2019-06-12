#include "experimental/opt/compares.hpp"
#include "experimental/opt/rule/convert.hpp"

#ifndef OPT_OPTIMIZE_HPP
#define OPT_OPTIMIZE_HPP

namespace opt
{

void replace_parents (const ade::ParentFinder& pfinder,
	ade::iTensor* source, ade::TensptrT target);

template <typename T>
std::vector<T> remove_duplicates (
	std::unordered_set<ade::iTensor*> priorities,
	std::vector<T> tens, const ade::ParentFinder& pfinder)
{
	if (tens.empty())
	{
		return {};
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
			// todo: mark parents as uninitialized, reinitialize entire graph, or uninitialize everything to begin with

			// inherit tags
			tag::move_tags(last.get(), cur.get());
			tag::erase(cur.get());
		}
		else
		{
			uniques.push_back(cur);
			last = cur;
		}
	}
	return uniques;
}

void optimize (ade::TensT roots, const rule::ConversionsT& conversions);

}

#endif // OPT_OPTIMIZE_HPP
