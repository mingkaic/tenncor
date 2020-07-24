///
/// derive.hpp
/// tenncor
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#include "tenncor/distr.hpp"

#ifndef TENNCOR_DERIVE_HPP
#define TENNCOR_DERIVE_HPP

namespace tcr
{

/// Derive root with respect to target and optimized
template <typename T>
eteq::ETensorsT<T> derive (
	eteq::ETensor<T> root,
	const eteq::ETensorsT<T>& targets)
{
	auto root_ctx = root.get_context();
	if (nullptr == root_ctx)
	{
		teq::fatal("root context is null");
	}

	if (std::any_of(targets.begin(), targets.end(),
		[&](const eteq::ETensor<T>& target)
		{
			return root_ctx != target.get_context();
		}))
	{
		teq::fatalf(
			"some target contexts don't match root %s context",
			root->to_string().c_str());
	}

	eteq::DerivativeFuncs<T> builder;

	if (auto deval = get_distreval(root_ctx))
	{
		// only look at reachable refs
		auto rrefs = distr::reachable_refs(teq::TensT{root.get()});
		if (rrefs.size() > 0)
		{
			distr::DRefptrSetT refs;
			auto owners = teq::track_owners({root});
			for (auto ref : rrefs)
			{
				refs.emplace(std::dynamic_pointer_cast<
					distr::iDistRef>(owners[ref].lock()));
			}
			teq::TensMapT<teq::TensptrsT> grads = {
				{root.get(), {builder.get_const_one(root->shape())}}
			};
			teq::TensptrSetT targset(targets.begin(), targets.end());

			auto tgrads = distrib_derive(
				grads, deval, {root}, targset, refs);

			size_t n = targets.size();
			eteq::ETensorsT<T> results;
			results.reserve(n);
			for (size_t i = 0; i < n; ++i)
			{
				auto target = targets[i];
				results.push_back(eteq::ETensor<T>(
					tgrads[target.get()], root.get_context()));
			}
			return results;
		}
	}

	teq::TensptrsT targs(targets.begin(), targets.end());
	teq::TensptrsT derivatives = teq::derive(root, targs, builder);
	eteq::ETensorsT<T> out;
	out.reserve(derivatives.size());
	std::transform(derivatives.begin(), derivatives.end(),
		std::back_inserter(out),
		[&root](teq::TensptrT tens)
		{
			return eteq::ETensor<T>(tens, root.get_context());
		});
	return out;
}

}

#endif // TENNCOR_DERIVE_HPP
