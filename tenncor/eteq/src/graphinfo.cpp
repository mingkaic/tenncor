#include "tenncor/eteq/graphinfo.hpp"

#ifdef ETEQ_GRAPHINFO_HPP

namespace eteq
{

const std::string graph_key = "graphinfo";

void set_graphinfo (GraphInfo* gi, global::CfgMapptrT ctx)
{
	ctx->rm_entry(graph_key);
	if (nullptr != gi)
	{
		ctx->template add_entry<GraphInfo>(graph_key,
			[=]{ return gi; });
	}
}

GraphInfo& get_graphinfo (const global::CfgMapptrT& ctx)
{
	auto gi = static_cast<GraphInfo*>(
		ctx->get_obj(graph_key));
	if (nullptr == gi)
	{
		gi = new GraphInfo();
		set_graphinfo(gi, ctx);
	}
	return *gi;
}

TensIdptrT TensIdentity::build (teq::TensptrT tens, const global::CfgMapptrT& ctx)
{
	bool hastens = nullptr != tens;
	if (nullptr == ctx && hastens)
	{
		global::fatalf("unsupported null context with non-null tensor %p", tens.get());
	}
	if (hastens)
	{
		auto& graphinfo = get_graphinfo(ctx);
		if (auto existing = graphinfo.get(tens))
		{
			return existing;
		}
		auto id = graphinfo.build_id();
		TensIdptrT out(new TensIdentity(id, ctx));
		graphinfo.emplace(tens, out);
		return out;
	}
	return nullptr;
}
	
TensIdentity::~TensIdentity (void)
{
	cleanup();
}

teq::TensptrT TensIdentity::get_tensor (void)
{
	return graphinfo_->get(this);
}
	
std::string TensIdentity::get_id (void) const
{
	return id_;
}

TensIdentity::TensIdentity (const std::string& id, const global::CfgMapptrT& ctx) : ctx_(ctx), id_(id)
{
	assert(nullptr != ctx); // logically guarded by static builder
	graphinfo_ = &get_graphinfo(ctx);
}

void TensIdentity::cleanup (void)
{
	if (false == ctx_.expired() && nullptr != graphinfo_)
	{
		ctx_ = global::CfgMapptrT(nullptr);
		graphinfo_->erase(this);
		graphinfo_ = nullptr;
	}
}

}

#endif
