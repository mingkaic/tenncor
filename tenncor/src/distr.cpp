
#include "tenncor/distr.hpp"

#ifdef TENNCOR_DISTR_HPP

namespace tcr
{

static distr::DRefptrSetT filter_reachable (
	distr::iDistrEvaluator* eval,
	const std::string& cid,
	const distr::DRefptrSetT& srcs,
	const teq::TensptrSetT& dests)
{
	return srcs;
}

static void remote_derive (
	teq::TensMapT<teq::TensptrsT>& grads,
	distr::iDistrEvaluator* eval,
	const std::string& cid,
	const distr::DRefptrSetT& roots,
	const teq::TensptrSetT& targets)
{
	//
}

distr::DEvalptrT make_distreval (
	ppconsul::Consul& consul, size_t port,
	const std::string& service,
	const std::string& id,
	const distr::ClientConfig& cfg)
{
	return std::make_shared<distr::DistrEvaluator>(
		distrib_derive, consul, port, service, id, cfg);
}

distr::iDistrEvaluator* get_distreval (eteq::ECtxptrT ctx)
{
	if (nullptr == ctx)
	{
		teq::fatal("cannot lookup references from null context");
	}
	return dynamic_cast<distr::iDistrEvaluator*>(ctx->eval_.get());
}

#define MAKE_DERFUNC(REAL_TYPE)\
builder = new eteq::DerivativeFuncs<REAL_TYPE>();

teq::TensMapT<teq::TensptrT> distrib_derive (
	teq::GradMapT& grads,
	distr::iDistrEvaluator* eval,
	const teq::TensptrSetT& roots,
	const teq::TensptrSetT& targets,
	const distr::DRefptrSetT& refs)
{
	if (roots.empty())
	{
		return {};
	}

	teq::iDerivativeFuncs* builder;
	auto dtype = (*roots.begin())->get_meta().type_code();
	TYPE_LOOKUP(MAKE_DERFUNC, dtype);

	if (refs.empty())
	{
		teq::partial_derive(grads, roots, targets, *builder);
	}
	else
	{
		teq::TensSetT refset;
		estd::StrMapT<distr::DRefptrSetT> remotes;
		for (auto ref : refs)
		{
			refset.emplace(ref.get());
			remotes[ref->cluster_id()].emplace(ref);
		}

		teq::TensptrSetT locals;
		for (auto root : roots)
		{
			if (false == estd::has(refset, root.get()))
			{
				locals.emplace(root);
			}
		}
		// filter target references by target reachability
		auto alltargs = targets;
		estd::StrMapT<distr::DRefptrSetT> rremotes;
		for (auto node : remotes)
		{
			auto cid = node.first;
			auto reachables = filter_reachable(
				eval, cid, node.second, targets);
			rremotes.emplace(cid, reachables);
			alltargs.insert(reachables.begin(), reachables.end());
		}
		// populate grads by local gradients
		if (locals.size() > 0)
		{
			teq::partial_derive(grads, locals, alltargs, *builder);
		}
		// then make remote calls
		for (auto node : rremotes)
		{
			remote_derive(grads, eval,
				node.first, node.second, targets);
		}
	}

	teq::TensMapT<teq::TensptrT> out;
	for (auto target : targets)
	{
		teq::TensptrT tens;
		teq::TensptrsT tgrads;
		if (estd::get(tgrads, grads, target.get()) && tgrads.size() > 0)
		{
			tens = tgrads.size() == 1 ?
				tgrads.front() : builder->add(tgrads);
		}
		else
		{
			tens = builder->get_const_zero(target->shape());
		}
		out.emplace(target.get(), tens);
	}
	delete builder;
	return out;
}

#undef MAKE_DERFUNC

}

#endif
