///
/// derive.hpp
/// tenncor
///
/// Purpose:
/// Implement eteq gradient definition for supported operations
///

#ifndef TENNCOR_DERIVE_HPP
#define TENNCOR_DERIVE_HPP

#include "tenncor/distr.hpp"

namespace tcr
{

template <typename T>
eteq::ETensorsT<T> derive_with_manager (
	distr::iDistrManager& mgr,
	eteq::ETensor<T> root,
	const eteq::ETensorsT<T>& targets)
{
	eteq::DerivativeFuncs<T> builder;
	teq::TensMapT<teq::TensptrsT> grads = {
		{root.get(), {builder.get_const_one(root->shape())}}
	};
	teq::TensSetT targset;
	std::transform(targets.begin(), targets.end(),
		std::inserter(targset, targset.end()),
		[](const eteq::ETensor<T>& etens)
		{
			return etens.get();
		});
	auto tgrads = distr::get_opsvc(mgr).derive(
		grads, {root}, distr::op::BackpropMeta{targset}, builder);

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

/// Derive root with respect to target and optimized
template <typename T>
eteq::ETensorsT<T> derive (
	eteq::ETensor<T> root,
	const eteq::ETensorsT<T>& targets)
{
	auto root_ctx = root.get_context();
	if (nullptr == root_ctx)
	{
		global::fatal("root context is null");
	}

	if (std::any_of(targets.begin(), targets.end(),
		[&](const eteq::ETensor<T>& target)
		{
			return root_ctx != target.get_context();
		}))
	{
		global::fatalf(
			"some target contexts don't match root %s context",
			root->to_string().c_str());
	}

	if (auto mgr = get_distrmgr(root_ctx))
	{
		return derive_with_manager(*mgr, root, targets);
	}

	eteq::DerivativeFuncs<T> builder;
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
