
#include "distrib/distrib.hpp"

#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

namespace tcr
{

void set_distrmgr (distr::iDistrMgrptrT mgr,
	global::CfgMapptrT ctx = global::context());

distr::iDistrManager* get_distrmgr (
	const global::CfgMapptrT& ctx = global::context());

template <typename T>
std::string expose_node (const eteq::ETensor<T>& etens)
{
	auto mgr = get_distrmgr(etens.get_context());
	return mgr->get_io().expose_node(etens);
}

template <typename T>
std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor<T> etens)
{
	auto mgr = get_distrmgr(etens.get_context());
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
		global::fatal(err->to_string());
	}
	return out;
}

template <typename T>
eteq::ETensor<T> try_lookup_node (
	error::ErrptrT& err, const std::string& id,
	const global::CfgMapptrT& ctx = global::context())
{
	auto mgr = get_distrmgr(ctx);
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
	const global::CfgMapptrT& ctx = global::context())
{
	error::ErrptrT err = nullptr;
	auto out = try_lookup_node<T>(err, id, ctx);
	if (nullptr != err)
	{
		global::fatal(err->to_string());
	}
	return out;
}

}

#endif // TENNCOR_DISTR_HPP
