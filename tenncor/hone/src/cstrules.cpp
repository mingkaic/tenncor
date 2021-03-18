#include "tenncor/hone/cstrules.hpp"

#ifdef HONE_CSTRULES_HPP

namespace hone
{

#define _CHOOSE_CST_TARGETTYPE(REALTYPE)\
if (auto sinfo = eigen::sparse_info(*func)){\
	out = eteq::make_constant_tensor<REALTYPE>((REALTYPE*)data, *sinfo, func->shape());\
}else{\
	out = eteq::make_constant_tensor<REALTYPE>((REALTYPE*)data, func->shape());\
}

teq::TensptrT constantize (teq::iTensor* func, const global::CfgMapptrT& ctx)
{
	eigen::Device device;
	teq::get_eval(ctx).evaluate(device, {func});
	void* data = func->device().data();
	auto outtype = (egen::_GENERATED_DTYPE) func->get_meta().type_code();
	teq::TensptrT out;
	TYPE_LOOKUP(_CHOOSE_CST_TARGETTYPE, outtype);
	return out;
}

#undef _CHOOSE_CST_TARGETTYPE

void generate_cstrules (opt::OptRulesT& rules,
	const opt::GraphInfo& graph, global::CfgMapptrT context)
{
	types::StrUMapT<std::unordered_set<size_t>> branches;
	for (const auto& owner : graph.get_owners())
	{
		if (auto f = dynamic_cast<const teq::iFunctor*>(owner.first))
		{
			branches[f->to_string()].emplace(f->get_args().size());
		}
	}
	if (branches.empty())
	{
		// for some reason...
		return;
	}
	google::protobuf::RepeatedPtrField<query::Node> srcs;
	for (auto& branch : branches)
	{
		for (size_t bfactor : branch.second)
		{
			if (branch.first != egen::name_op(egen::IDENTITY)) // todo: move to generated output
			{
				query::Node* src = srcs.Add();
				get_cstsource(*src, branch.first, bfactor);
			}
		}
	}
	rules.push_back(opt::OptRule{srcs,
		std::make_shared<ConstantTarget>(graph, context)});
}

}

#endif
