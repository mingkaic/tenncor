/// optimize.hpp
/// eteq
///
/// Purpose:
/// Extend optimization module by defining ETEQ node parsing
///

#ifndef HONE_OPTIMIZE_HPP
#define HONE_OPTIMIZE_HPP

#include <fstream>

#include "tenncor/hone/duplicates.hpp"
#include "tenncor/hone/cstrules.hpp"
#include "tenncor/hone/matchain.hpp"
#include "tenncor/hone/target.hpp"

namespace hone
{

using GenCstF = std::function<void(opt::OptRulesT&,const opt::GraphInfo&)>;

teq::TensptrsT optimize (teq::TensptrsT roots,
	const opt::Optimization& pb_opt, GenCstF gen_cst =
	[](opt::OptRulesT& rules, const opt::GraphInfo& ginfo)
	{
		generate_cstrules(rules, ginfo);
	});

teq::TensptrsT optimize (teq::TensptrsT roots,
	std::istream& rulestr, GenCstF gen_cst =
	[](opt::OptRulesT& rules, const opt::GraphInfo& ginfo)
	{
		generate_cstrules(rules, ginfo);
	});

/// Apply optimization to graph roots in context registry
void optimize (std::string filename,
	const global::CfgMapptrT& ctx = global::context());

}

#endif // HONE_OPTIMIZE_HPP
