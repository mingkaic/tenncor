#include "internal/eigen/memory.hpp"

#ifdef EIGEN_MEMORY_HPP

namespace eigen
{

const std::string memory_key = "runtime_memory";

void set_runtime (RTMemptrT mem, global::CfgMapptrT ctx)
{
	ctx->rm_entry(memory_key);
	if (mem)
	{
		ctx->template add_entry<RTMemptrT>(memory_key,
		[=]{ return new RTMemptrT(mem); });
	}
}

RTMemptrT get_runtime (const global::CfgMapptrT& ctx)
{
	auto rt = static_cast<RTMemptrT*>(ctx->get_obj(memory_key));
	if (nullptr != rt)
	{
		return *rt;
	}
	auto rtmem = std::make_shared<RuntimeMemory>();
	set_runtime(rtmem, ctx);
	return rtmem;
}

}

#endif // EIGEN_MEMORY_HPP
