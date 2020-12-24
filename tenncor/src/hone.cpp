#include "tenncor/hone.hpp"

#ifdef TENNCOR_HONE_HPP

namespace tcr
{

static teq::TensptrsT optimize (
	const teq::TensptrsT& roots,
	std::istream& json_in,
	const global::CfgMapptrT& ctx)
{
	if (roots.empty())
	{
		return {};
	}
	if (auto mgr = get_distrmgr(ctx))
	{
		opt::Optimization pb_opt;
		opt::json2optimization(pb_opt, json_in);
		return distr::get_hosvc(*mgr).optimize(roots, pb_opt);
	}
	return hone::optimize(roots, json_in);
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
	auto outroots = optimize(inroots, rulefile, ctx);
	assert(inroots.size() == outroots.size());
	auto& graphinfo = eteq::get_graphinfo(ctx);
	for (size_t i = 0, n = inroots.size(); i < n; ++i)
	{
		graphinfo.replace(inroots[i], outroots[i]);
	}
}

}

#endif
