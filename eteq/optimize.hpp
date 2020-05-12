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
	opt::GraphInfo graph(teq::TensptrsT(roots.begin(), roots.end()));
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
	auto oroots = graph.get_roots();
	for (size_t i = 0, n = roots.size(); i < n; ++i)
	{
		roots[i] = eteq::ETensor<T>(oroots[i], *roots[i].get_registry());
	}
}

/// Apply optimization to graph roots tracked by session
template <typename T>
void optimize (eteq::ETensContext& context, std::string filename)
{
	std::ifstream rulefile(filename);
	auto& reg = context.registry_;
	teq::TensptrSetT roots;
	for (auto& rpairs : reg)
	{
		roots.emplace(rpairs.second);
	}

	teq::TensMapT<teq::TensptrT> changed;
	{
		teq::TensptrsT rootvec(roots.begin(), roots.end());
		eteq::ETensorsT<T> order;
		order.reserve(rootvec.size());
		std::transform(rootvec.begin(), rootvec.end(),
			std::back_inserter(order),
			[&](teq::TensptrT root)
			{
				return eteq::ETensor<T>(root, reg);
			});
		optimize<T>(order, rulefile);
		for (size_t i = 0, n = rootvec.size(); i < n; ++i)
		{
			changed.emplace(rootvec[i].get(), order[i]);
		}
	}

	for (auto& rpairs : reg)
	{
		rpairs.second = changed[rpairs.second.get()];
	}
}

}

#endif // ETEQ_OPT_HPP
