
#include "distrib/distrib.hpp"
#include "eteq/eteq.hpp"

#ifndef TENNCOR_DISTR_HPP
#define TENNCOR_DISTR_HPP

namespace tcr
{

#define _MAKE_DERFUNC(REAL_TYPE)\
builder = new eteq::DerivativeFuncs<REAL_TYPE>();

struct TenncorManager final : public distr::DistManager
{
	template<typename... ARGS>
	TenncorManager (ARGS&&... args) :
		DistManager(std::forward<ARGS>(args)...) {}

	/// Implementation of iDistManager
	teq::TensMapT<teq::TensptrT> derive (
		teq::GradMapT& grads,
		const teq::TensptrSetT& roots,
		const teq::TensptrSetT& targets) override
	{
		if (roots.empty())
		{
			return {};
		}

		teq::iDerivativeFuncs* builder;
		auto dtype = (*roots.begin())->get_meta().type_code();
		TYPE_LOOKUP(_MAKE_DERFUNC, dtype);

		// only look at reachable refs
		auto refs = distr::reachable_refs(roots);
		if (refs.empty())
		{
			teq::partial_derive(grads, roots, targets, *builder);
		}
		else
		{
			estd::StrMapT<estd::StrSetT> remotes;
			for (auto ref : refs)
			{
				remotes[ref->cluster_id()].emplace(ref->node_id());
			}

			teq::TensptrSetT locals;
			teq::TensSetT refset(refs.begin(), refs.end());
			for (auto root : roots)
			{
				if (false == estd::has(refset, root.get()))
				{
					locals.emplace(root);
				}
			}

			estd::StrSetT targids;
			for (auto target : targets)
			{
				// targets that are not exposed can't be referenced remotely
				if (auto id = lookup_id(target.get()))
				{
					targids.emplace(*id);
				}
			}
			// filter target references by target reachability
			auto local_target = targets;
			for (auto& remote : remotes)
			{
				error::ErrptrT err = nullptr;
				remote.second = remote_find_reachable(err,
					remote.first, remote.second, targids);
				if (nullptr != err)
				{
					teq::fatal(err->to_string());
				}
				for (auto reachable : remote.second)
				{
					auto ref = lookup_node(err, reachable);
					if (nullptr != err)
					{
						teq::fatal(err->to_string());
					}
					local_target.emplace(ref);
				}
			}
			// populate grads by local gradients
			if (locals.size() > 0)
			{
				teq::partial_derive(grads, locals, local_target, *builder);
			}
			// then make remote calls
			for (auto remote : remotes)
			{
				remote_derive(grads, remote.first, remote.second, targids);
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

private:
	void remote_derive (
		teq::TensMapT<teq::TensptrsT>& grads,
		const std::string& peer_id,
		const estd::StrSetT& roots,
		const estd::StrSetT& targets)
	{
		//
	}
};

#undef _MAKE_DERFUNC

const std::string distmgr_key = "distmanager";

void set_distmgr (distr::iDistMgrptrT mgr);

template <typename CTX> // concept CTX is TensContext pointer
void set_distmgr (distr::iDistMgrptrT mgr, CTX ctx)
{
	ctx->owners_.erase(distmgr_key);
	if (nullptr != mgr)
	{
		ctx->owners_.insert(std::pair<std::string,eigen::OwnerptrT>{
			distmgr_key, std::make_unique<distr::ManagerOwner>(mgr)});
	}
	ctx->eval_ = std::make_shared<distr::DistEvaluator>(mgr.get());
}

template <typename CTX> // todo: concept CTX is TensContext pointer
distr::iDistManager* get_distmgr (const CTX& ctx)
{
	if (nullptr == ctx)
	{
		teq::fatal("cannot lookup references from null context");
	}
	if (false == estd::has(ctx->owners_, distmgr_key))
	{
		return nullptr;
	}
	return static_cast<distr::iDistManager*>(
		ctx->owners_.at(distmgr_key)->get_raw());
}

template <typename T>
void expose_node (const eteq::ETensor<T>& etens)
{
	auto mgr = get_distmgr(etens.get_context());
	mgr->expose_node(etens);
}

template <typename T>
std::string try_lookup_id (error::ErrptrT& err, eteq::ETensor<T> etens)
{
	auto ctx = etens.get_context();
	auto mgr = get_distmgr(ctx);
	if (nullptr == mgr)
	{
		err = error::error(
			"can only find reference ids using iDistManager");
		return "";
	}
	auto opt_id = mgr->lookup_id(etens.get());
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
			"can only find references using iDistManager");
		return eteq::ETensor<T>();
	}
	return eteq::ETensor<T>(mgr->lookup_node(err, id), ctx);
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
