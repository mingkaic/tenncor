
#include "distrib/distrib.hpp"
#include "eteq/eteq.hpp"

#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

namespace tcr
{

distr::DEvalptrT make_distreval (
	ppconsul::Consul& consul, size_t port,
	const std::string& service,
	const std::string& id = "",
	const distr::ClientConfig& cfg = distr::ClientConfig());

distr::iDistrEvaluator* get_distreval (eteq::ECtxptrT ctx);

teq::TensMapT<teq::TensptrT> distrib_derive (
	teq::GradMapT& grads,
	distr::iDistrEvaluator* eval,
	const teq::TensptrSetT& roots,
	const teq::TensptrSetT& targets,
	const distr::DRefptrSetT& refs);

template <typename T>
void expose_node (const eteq::ETensor<T>& etens)
{
	auto eval = get_distreval(etens.get_context());
	eval->expose_node(etens);
}

template <typename T>
std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor<T> etens)
{
	auto ctx = etens.get_context();
	auto eval = get_distreval(ctx);
	if (nullptr == eval)
	{
		err = error::error(
			"cannot only find reference ids using iDistrEvaluator");
		return "";
	}
	auto opt_id = eval->lookup_id(etens);
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
	eteq::ECtxptrT ctx = eteq::global_context())
{
	auto eval = get_distreval(ctx);
	if (nullptr == eval)
	{
		err = error::error(
			"cannot only find references using iDistrEvaluator");
		return eteq::ETensor<T>();
	}
	return eteq::ETensor<T>(eval->lookup_node(err, id), ctx);
}

template <typename T>
eteq::ETensor<T> lookup_node (const std::string& id,
	eteq::ECtxptrT ctx = eteq::global_context())
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
