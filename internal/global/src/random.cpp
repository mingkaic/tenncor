#include "internal/global/random.hpp"

#ifdef GLOBAL_RANDOM_HPP

namespace global
{

const std::string generator_key = "generator";

void set_generator (GenPtrT gen, CfgMapptrT ctx)
{
	ctx->rm_entry(generator_key);
	if (gen)
	{
		ctx->template add_entry<GenPtrT>(generator_key,
		[=]{ return new GenPtrT(gen); });
	}
}

GenPtrT get_generator (const CfgMapptrT& ctx)
{
	auto gen = static_cast<GenPtrT*>(ctx->get_obj(generator_key));
	if (nullptr != gen)
	{
		return *gen;
	}
	auto rgen = std::make_shared<Randomizer>();
	set_generator(rgen, ctx);
	return rgen;
}

void seed (size_t s, const CfgMapptrT& ctx)
{
	if (auto gen = dynamic_cast<iRandGenerator*>(get_generator(ctx).get()))
	{
		gen->seed(s);
	}
}

}

#endif
