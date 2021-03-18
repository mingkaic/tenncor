
#include "tenncor/eteq.hpp"

#ifdef TENNCOR_ETEQ_HPP

namespace tcr
{

eteq::ETensorsT derive_with_manager (
	distr::iDistrManager& mgr,
	eteq::ETensor root,
	const eteq::ETensorsT& targets)
{
	eteq::DerivativeFuncs builder;
	teq::TensMapT<teq::TensptrsT> grads = {
		{root.get(), {builder.get_const_one(*root)}}
	};
	teq::TensSetT targset;
	teq::multi_get(targets.begin(), targets.end(),
		std::inserter(targset, targset.end()));
	auto tgrads = distr::get_opsvc(mgr).derive(
		grads, {root}, distr::op::BackpropMeta{targset});

	size_t n = targets.size();
	eteq::ETensorsT results;
	results.reserve(n);
	for (size_t i = 0; i < n; ++i)
	{
		auto target = targets[i];
		results.push_back(eteq::ETensor(
			tgrads[target.get()], root.get_context()));
	}
	return results;
}

eteq::ETensorsT derive (eteq::ETensor root, const eteq::ETensorsT& targets)
{
	auto root_ctx = root.get_context();
	if (nullptr == root_ctx)
	{
		global::fatal("root context is null");
	}

	if (std::any_of(targets.begin(), targets.end(),
		[&](const eteq::ETensor& target)
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

	eteq::DerivativeFuncs builder;
	teq::TensptrsT targs(targets.begin(), targets.end());
	//teq::TensptrsT derivatives = teq::backprop(root, targs, builder);
    teq::TensptrsT derivatives = teq::derive(root, targs, builder);
	eteq::ETensorsT out;
	out.reserve(derivatives.size());
	std::transform(derivatives.begin(), derivatives.end(),
	std::back_inserter(out),
	[&root,&root_ctx](teq::TensptrT tens)
	{
		return eteq::ETensor(tens, root_ctx);
	});
	return out;
}

}

#endif
