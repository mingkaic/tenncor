/// optimize.hpp
/// eteq
///
/// Purpose:
/// Extend optimization module by defining ETEQ node parsing
///

#include <fstream>

#include "opt/parse.hpp"
#include "opt/apply.hpp"

#include "eteq/duplicates.hpp"
#include "eteq/cstrules.hpp"
#include "eteq/target.hpp"

#ifndef ETEQ_OPT_HPP
#define ETEQ_OPT_HPP

namespace eteq
{

const size_t convert_round_limit = 50;

template <typename T>
void optimize (eteq::ETensorsT<T>& roots, std::istream& rulestr)
{
	opt::OptRulesT rules;
	opt::UnindexedGraph gbase(teq::TensptrsT(roots.begin(), roots.end()));
	merge_dups<T>(gbase); // remove duplicates to reduce search space

	opt::GraphInfo graph(gbase);
	eteq::TargetFactory<T> impl_factory(graph);
	eteq::generate_cstrules<T>(rules, graph); // populate with constant rules
	opt::json_parse(rules, rulestr, impl_factory);
	bool converted = true;
	for (size_t i = 0; converted && i < convert_round_limit; ++i)
	{
		converted = opt::optimize(graph, rules);
	}
	// apply new roots
	auto altered_roots = graph.get_roots();
	for (size_t i = 0, n = roots.size(); i < n; ++i)
	{
		if (altered_roots[i] != roots[i].get())
		{
			roots[i] = graph.get_owner(altered_roots[i]);
		}
	}
}

/// Apply optimization to graph roots tracked by session
template <typename T>
void optimize (teq::iSession& sess, std::string filename)
{
	std::ifstream rulefile(filename);
	teq::TensptrSetT tracked_set = sess.get_tracked();
	eteq::ETensorsT<T> tracked(tracked_set.begin(), tracked_set.end());
	optimize<T>(tracked, rulefile);
	sess.clear();
	sess.track(teq::TensptrsT(tracked.begin(), tracked.end()));
}

}

#endif // ETEQ_OPT_HPP
