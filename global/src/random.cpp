#include "global/random.hpp"

#ifdef GLOBAL_RANDOM_HPP

namespace global
{

const std::string rengine_key = "rengine";

const std::string uengine_key = "uengine";

void set_randengine (RandEngineT* reg, global::CfgMapptrT ctx)
{
	ctx->rm_entry(rengine_key);
	if (reg)
	{
		ctx->template add_entry<RandEngineT>(rengine_key,
			[=](){ return reg; });
	}
}

RandEngineT& get_randengine (global::CfgMapptrT ctx)
{
	auto reg = static_cast<RandEngineT*>(
		ctx->get_obj(rengine_key));
	if (nullptr == reg)
	{
		reg = new RandEngineT();
		set_randengine(reg, ctx);
	}
	return *reg;
}

void set_uuidengine (BoostEngineT* reg, global::CfgMapptrT ctx)
{
	ctx->rm_entry(uengine_key);
	if (reg)
	{
		ctx->template add_entry<BoostEngineT>(uengine_key,
			[=](){ return reg; });
	}
}

BoostEngineT& get_uuidengine (global::CfgMapptrT ctx)
{
	auto reg = static_cast<BoostEngineT*>(
		ctx->get_obj(uengine_key));
	if (nullptr == reg)
	{
		reg = new BoostEngineT();
		set_uuidengine(reg, ctx);
	}
	return *reg;
}

}

#endif
