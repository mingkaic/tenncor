#include "tenncor/opt/optimize.hpp"

#ifdef TENNCOR_OPT_OPTIMIZE_HPP

namespace eteq
{

static const size_t convert_round_limit = 50;

teq::TensptrsT optimize (teq::TensptrsT roots,
	std::istream& rulestr, GenCstF gen_cst)
{
	opt::OptRulesT rules;
	opt::GraphInfo graph(roots);
	merge_dups(graph); // remove duplicates to reduce search space

	TargetFactory impl_factory(graph);
	if (gen_cst)
	{
		gen_cst(rules, graph); // populate with constant rules
	}
	opt::json_parse(rules, rulestr, impl_factory);
	bool converted = true;
	for (size_t i = 0; converted && i < convert_round_limit; ++i)
	{
		converted = opt::optimize(graph, rules);
	}
	// apply new roots
	return graph.get_roots();
}

teq::TensptrsT optimize (teq::TensptrsT roots, std::istream& rulestr)
{
	return optimize(roots, rulestr,
		[](opt::OptRulesT& rule, const opt::GraphInfo& graph)
		{ generate_cstrules(rule, graph); });
}

void optimize (std::string filename, const global::CfgMapptrT& ctx)
{
	std::ifstream rulefile(filename);
	auto& reg = get_reg(ctx);
	teq::TensptrSetT roots;
	for (auto& rpairs : reg)
	{
		roots.emplace(rpairs.second);
	}

	teq::OwnMapT changed;
	teq::TensptrsT inroots(roots.begin(), roots.end());
	auto outroots = optimize(inroots, rulefile,
		[&ctx](opt::OptRulesT& rules, const opt::GraphInfo& graph)
		{
			generate_cstrules(rules, graph, ctx);
		});
	assert(inroots.size() == outroots.size());
	for (size_t i = 0, n = inroots.size(); i < n; ++i)
	{
		changed.emplace(inroots[i].get(), outroots[i]);
	}

	for (auto& rpairs : reg)
	{
		rpairs.second = changed[rpairs.second.get()];
	}
}

}

#endif
