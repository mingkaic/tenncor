
#include "eteq/etens.hpp"

#ifdef ETEQ_ETENS_HPP

namespace eteq
{

const std::string reg_key = "tens_registry";

void set_reg (TensRegistryT* reg, global::CfgMapptrT ctx)
{
	ctx->rm_entry(reg_key);
	if (reg)
	{
		ctx->template add_entry<TensRegistryT>(reg_key,
			[=](){ return reg; });
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

}

#endif
