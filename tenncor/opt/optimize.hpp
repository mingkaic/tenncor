/// optimize.hpp
/// eteq
///
/// Purpose:
/// Extend optimization module by defining ETEQ node parsing
///

#ifndef TENNCOR_OPT_OPTIMIZE_HPP
#define TENNCOR_OPT_OPTIMIZE_HPP

#include <fstream>

#include "tenncor/opt/duplicates.hpp"
#include "tenncor/opt/cstrules.hpp"
#include "tenncor/opt/target.hpp"

namespace eteq
{

using GenCstF = std::function<void(opt::OptRulesT&,const opt::GraphInfo&)>;

teq::TensptrsT optimize (teq::TensptrsT roots,
	std::istream& rulestr, GenCstF gen_cst);

teq::TensptrsT optimize (teq::TensptrsT roots, std::istream& rulestr);

/// Apply optimization to graph roots in context registry
void optimize (std::string filename,
	const global::CfgMapptrT& ctx = global::context());

}

#endif // TENNCOR_OPT_OPTIMIZE_HPP
