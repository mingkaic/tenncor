#include "tenncor/hone/cstrules.hpp"

#ifdef HONE_CSTRULES_HPP

namespace hone
{

void generate_cstrules (opt::OptRulesT& rules,
	const opt::GraphInfo& graph, global::CfgMapptrT context)
{
	std::unordered_map<std::string,std::unordered_set<size_t>> branches;
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
			query::Node* src = srcs.Add();
			get_cstsource(*src, branch.first, bfactor);
		}
	}
	rules.push_back(opt::OptRule{srcs,
		std::make_shared<ConstantTarget>(graph, context)});
}

}

#endif
