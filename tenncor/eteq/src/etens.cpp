
#include "tenncor/eteq/etens.hpp"

#ifdef ETEQ_ETENS_HPP

namespace eteq
{

const std::string reg_key = "tens_registry";

teq::TensptrsT to_tensors (const ETensorsT& etensors)
{
	teq::TensptrsT tensors;
	tensors.reserve(etensors.size());
	std::transform(etensors.begin(), etensors.end(),
		std::back_inserter(tensors),
		[](ETensor etens)
		{
			return (teq::TensptrT) etens;
		});
	return tensors;
}

void set_reg (TensRegistryT* reg, global::CfgMapptrT ctx)
{
	ctx->rm_entry(reg_key);
	if (nullptr != reg)
	{
		ctx->template add_entry<TensRegistryT>(reg_key,
			[=]{ return reg; });
	}
}

TensRegistryT& get_reg (const global::CfgMapptrT& ctx)
{
	auto reg = static_cast<TensRegistryT*>(
		ctx->get_obj(reg_key));
	if (nullptr == reg)
	{
		reg = new TensRegistryT();
		set_reg(reg, ctx);
	}
	return *reg;
}

void run (const ETensorsT& targets, teq::TensSetT ignored, size_t max_version)
{
	if (targets.empty())
	{
		return;
	}
	if (auto ctx = targets.front().get_context())
	{
		eigen::Device device(max_version);
		teq::TensSetT targset;
		for (const auto& etens : targets)
		{
			targset.emplace(etens.get());
		}
		teq::get_eval(ctx).evaluate(device, targset, ignored);
	}
}

}

#endif
