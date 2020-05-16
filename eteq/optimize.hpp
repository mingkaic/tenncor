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
teq::TensptrsT optimize (teq::TensptrsT roots, std::istream& rulestr)
{
	opt::OptRulesT rules;
	opt::GraphInfo graph(roots);
	merge_dups<T>(graph); // remove duplicates to reduce search space

	eteq::TargetFactory<T> impl_factory(graph);
	eteq::generate_cstrules<T>(rules, graph); // populate with constant rules
	opt::json_parse(rules, rulestr, impl_factory);
	bool converted = true;
	for (size_t i = 0; converted && i < convert_round_limit; ++i)
	{
		converted = opt::optimize(graph, rules);
	}
	// apply new roots
	return graph.get_roots();
}

/// Apply optimization to graph roots tracked by session
template <typename T>
void optimize (eteq::ECtxptrT context, std::string filename)
{
	std::ifstream rulefile(filename);
	auto& reg = context->registry_;
	teq::TensptrSetT roots;
	for (auto& rpairs : reg)
	{
		roots.emplace(rpairs.second);
	}

	teq::TensMapT<teq::TensptrT> changed;
	teq::TensptrsT inroots(roots.begin(), roots.end());
	auto outroots = optimize<T>(inroots, rulefile);
	assert(inroots.size() == outroots.size());
	for (size_t i = 0, n = inroots.size(); i < n; ++i)
	{
		changed.emplace(inroots[i].get(), outroots[i]);
	}

	for (auto& rpairs : reg)
	{
		rpairs.second = changed[rpairs.second.get()];
	}
	context->sess_->clear();
	context->sess_->track(outroots);
}

}

#endif // ETEQ_OPT_HPP
