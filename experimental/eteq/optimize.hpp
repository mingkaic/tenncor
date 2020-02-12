/// optimize.hpp
/// eteq
///
/// Purpose:
/// Extend optimization module by defining ETEQ node parsing
///

#include <fstream>

#include "experimental/opt/parse.hpp"
#include "experimental/opt/apply.hpp"

#include "experimental/eteq/cstrules.hpp"
#include "experimental/eteq/target.hpp"

#ifndef EXPERIMENTAL_ETEQ_OPT_HPP
#define EXPERIMENTAL_ETEQ_OPT_HPP

namespace eteq
{

template <typename T>
void optimize (eteq::ETensorsT<T>& roots, std::istream& rulestr)
{
	opt::OptRulesT rules;
	opt::GraphInfo graph(teq::TensptrsT(roots.begin(), roots.end()));

	eteq::TargetFactory<T> impl_factory(graph);
	eteq::generate_cstrules<T>(rules, graph); // populate with constant rules
	opt::json_parse(rules, rulestr, impl_factory);
	bool converted = true;
	while (converted)
	{
		converted = opt::optimize(graph, rules);
	}
	for (auto& root : roots)
	{
		if (estd::has(graph.changes_, root.get()))
		{
			root = graph.changes_.at(root.get());
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
	sess.track(tracked);
}

}

#endif // EXPERIMENTAL_ETEQ_OPT_HPP
