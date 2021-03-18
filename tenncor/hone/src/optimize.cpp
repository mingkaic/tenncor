#include "tenncor/hone/optimize.hpp"

#ifdef HONE_OPTIMIZE_HPP

namespace hone
{

using ApplyRulesF = std::function<void(opt::OptRulesT&,const TargetFactory&)>;

static const size_t convert_round_limit = 50;

static teq::TensptrsT general_optimize (
	teq::TensptrsT roots, GenCstF gen_cst, ApplyRulesF apply_rules)
{
	opt::OptRulesT rules;
	opt::GraphInfo graph(roots);
	merge_dups(graph); // remove duplicates to reduce search space

	TargetFactory impl_factory(graph);
	if (gen_cst)
	{
		gen_cst(rules, graph); // populate with constant rules
	}
	apply_rules(rules, impl_factory);

	bool converted = true;
	for (size_t i = 0; converted && i < convert_round_limit; ++i)
	{
		converted = opt::optimize(graph, rules);
	}

	matrix_chain(graph); // matrix chaining
	// apply new roots
	return graph.get_roots();
}

teq::TensptrsT optimize (teq::TensptrsT roots,
	const opt::Optimization& pb_opt, GenCstF gen_cst)
{
	return general_optimize(roots, gen_cst,
	[&pb_opt](opt::OptRulesT& rules, const TargetFactory& tfac)
	{
		opt::parse_optimization(rules, pb_opt, tfac);
	});
}

teq::TensptrsT optimize (teq::TensptrsT roots,
	std::istream& rulestr, GenCstF gen_cst)
{
	return general_optimize(roots, gen_cst,
	[&rulestr](opt::OptRulesT& rules, const TargetFactory& tfac)
	{
		opt::json_parse(rules, rulestr, tfac);
	});
}

void optimize (std::string filename, const global::CfgMapptrT& ctx)
{
	std::ifstream rulefile(filename);
	auto& reg = eteq::get_reg(ctx);
	teq::TensptrSetT roots;
	for (auto& rpairs : reg)
	{
		roots.emplace(rpairs.second->get_tensor());
	}

	teq::TensptrsT inroots(roots.begin(), roots.end());
	auto outroots = optimize(inroots, rulefile,
	[&ctx](opt::OptRulesT& rules, const opt::GraphInfo& ginfo)
	{
		generate_cstrules(rules, ginfo, ctx);
	});
	assert(inroots.size() == outroots.size());
	auto& graphinfo = eteq::get_graphinfo(ctx);
	for (size_t i = 0, n = inroots.size(); i < n; ++i)
	{
		graphinfo.replace(inroots[i], outroots[i]);
	}
}

}

#endif
