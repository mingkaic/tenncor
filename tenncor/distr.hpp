
#include "distrib/distrib.hpp"
#include "eteq/eteq.hpp"

#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

namespace tcr
{

const std::string distmgr_key = "DistrManager";

void set_distmgr (distr::iDistMgrptrT mgr);

template <typename CTX> // concept CTX is TensContext pointer
void set_distmgr (distr::iDistMgrptrT mgr, CTX ctx)
{
	ctx->owners_.erase(distmgr_key);
	if (nullptr != mgr)
	{
		ctx->owners_.insert(std::pair<std::string,eigen::OwnerptrT>{
			distmgr_key, std::make_unique<distr::ManagerOwner>(mgr)});
		ctx->eval_ = std::make_shared<distr::DistrEvaluator>(*mgr);
	}
	else
	{
		ctx->eval_ = std::make_shared<teq::Evaluator>();
	}
}

template <typename CTX> // todo: concept CTX is TensContext pointer
distr::iDistrManager* get_distmgr (const CTX& ctx)
{
	if (nullptr == ctx)
	{
		teq::fatal("cannot lookup references from null context");
	}
	if (false == estd::has(ctx->owners_, distmgr_key))
	{
		return nullptr;
	}
	return static_cast<distr::iDistrManager*>(
		ctx->owners_.at(distmgr_key)->get_raw());
}

template <typename T>
std::string expose_node (const eteq::ETensor<T>& etens)
{
	auto mgr = get_distmgr(etens.get_context());
	return mgr->get_io().expose_node(etens);
}

template <typename T>
std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor<T> etens)
{
	auto ctx = etens.get_context();
	auto mgr = get_distmgr(ctx);
	if (nullptr == mgr)
	{
		err = error::error(
			"can only find reference ids using DistrManager");
		return "";
	}
	auto opt_id = mgr->get_io().lookup_id(etens.get());
	if (false == bool(opt_id))
	{
		err = error::errorf("failed to find tensor %s",
			etens->to_string().c_str());
		return "";
	}
	return *opt_id;
}

template <typename T>
std::string lookup_id (eteq::ETensor<T> etens)
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_id(err, etens);
	if (nullptr != err)
	{
		teq::fatal(err->to_string());
	}
	return out;
}

template <typename T>
eteq::ETensor<T> try_lookup_node (
	error::ErrptrT& err, const std::string& id,
	eigen::CtxptrT ctx = eigen::global_context())
{
	auto mgr = get_distmgr(ctx.get());
	if (nullptr == mgr)
	{
		err = error::error(
			"can only find references using DistrManager");
		return eteq::ETensor<T>();
	}
	return eteq::ETensor<T>(mgr->get_io().lookup_node(err, id), ctx);
}

template <typename T>
eteq::ETensor<T> lookup_node (const std::string& id,
	eigen::CtxptrT ctx = eigen::global_context())
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_node<T>(err, id, ctx);
	if (nullptr != err)
	{
		teq::fatal(err->to_string());
	}
	return out;
}

}

#endif // TENNCOR_DISTR_HPP
