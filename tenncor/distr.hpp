
#include "distrib/distrib.hpp"
#include "eteq/eteq.hpp"

#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

namespace tcr
{

std::shared_ptr<distr::DistribSess> make_distrsess (
	ppconsul::Consul& consul, size_t port,
	const std::string& service,
	const std::string& id = "",
	const distr::ClientConfig& cfg = distr::ClientConfig());

distr::iDistribSess* get_distrsess (eteq::ECtxptrT ctx);

teq::TensMapT<teq::TensptrT> distrib_derive (
	teq::GradMapT& grads,
	distr::iDistribSess* sess,
	const teq::TensptrSetT& roots,
	const teq::TensptrSetT& targets,
	const distr::DRefptrSetT& refs);

template <typename T>
std::string try_lookup_id (err::ErrptrT& err, eteq::ETensor<T> etens)
{
	auto ctx = etens.get_context();
	auto sess = get_distrsess(ctx);
	if (nullptr == sess)
	{
		err = std::make_shared<err::ErrMsg>(
			"cannot only find reference ids using iDistribSess");
		return "";
	}
	auto opt_id = sess->lookup_id(etens);
	if (false == bool(opt_id))
	{
		err = std::make_shared<err::ErrMsg>(fmts::sprintf(
			"failed to find tensor %s", etens->to_string().c_str()));
		return "";
	}
	return *opt_id;
}

template <typename T>
std::string lookup_id (eteq::ETensor<T> etens)
{
	err::ErrptrT err = nullptr;
	auto out = try_lookup_id(err, etens);
	if (nullptr != err)
	{
		teq::fatal(err->to_string());
	}
	return out;
}

template <typename T>
eteq::ETensor<T> try_lookup_node (
	err::ErrptrT& err, const std::string& id,
	eteq::ECtxptrT ctx = eteq::global_context())
{
	auto sess = get_distrsess(ctx);
	if (nullptr == sess)
	{
		err = std::make_shared<err::ErrMsg>(
			"cannot only find references using iDistribSess");
		return eteq::ETensor<T>();
	}
	return eteq::ETensor<T>(sess->lookup_node(err, id), ctx);
}

template <typename T>
eteq::ETensor<T> lookup_node (const std::string& id,
	eteq::ECtxptrT ctx = eteq::global_context())
{
	err::ErrptrT err = nullptr;
	auto out = try_lookup_node<T>(err, id, ctx);
	if (nullptr != err)
	{
		teq::fatal(err->to_string());
	}
	return out;
}

}

#endif // TENNCOR_DISTR_HPP
