
#ifndef ETEQ_OPT_CSTRULES_HPP
#define ETEQ_OPT_CSTRULES_HPP

// Define rules for identifying and merging operators with only constant arguments

// 1. identity all valid branching factor for each opcode

// 2. create a source graph for each branching factor of each opcode

// 3. create a custom target producer for calculating constant values

/* source graphs should take the form:

{
	"op":{
		"opname":"<OPCODE>",
		"args":[
			{
				"leaf":{"usage":"constant"}
			},
			...
		],
		"capture":"root"
	}
}

Symbols map has {"root":<op with constant args>}
Target operates on "root"
*/

#include "eigen/device.hpp"

#include "opt/graph.hpp"
#include "opt/rule.hpp"
#include "opt/target.hpp"

#include "eteq/make.hpp"

namespace eteq
{

// custom target for calculating constant values
template <typename T>
struct ConstantTarget final : public opt::iTarget
{
	ConstantTarget (const opt::GraphInfo& graph) : graph_(&graph) {}

	teq::TensptrT convert (const query::SymbMapT& candidates) const override
	{
		auto root = candidates.at("root");
		teq::Session sess = eigen::get_session();
		sess.track({graph_->get_owner(root)});
		sess.update_target({root});
		T* data = (T*) root->device().data();
		return make_constant(data, root->shape());
	}

	const opt::GraphInfo* graph_;
};

// source graph for certain branching factor of certain operator
static query::ConditionT get_cstsource (std::string opname, size_t nbranch)
{
	query::ConditionT node = std::make_shared<query::Node>();
	query::Operator* op = node->mutable_op();
	op->set_opname(opname);
	op->set_capture("root");
	for (size_t i = 0; i < nbranch; ++i)
	{
		query::Node* arg = op->add_args();
		query::Leaf* leaf = arg->mutable_leaf();
		leaf->set_usage(teq::get_usage_name(teq::Usage::IMMUTABLE));
	}
	return node;
}

// gather branching factor of all operators,
// then append rule converting constant source to constant target
template <typename T>
void generate_cstrules (opt::OptRulesT& rules, const opt::GraphInfo& graph)
{
	std::unordered_map<std::string,std::unordered_set<size_t>> branches;
	for (const auto& owner : graph.get_owners())
	{
		if (auto f = dynamic_cast<const teq::iFunctor*>(owner.first))
		{
			branches[f->to_string()].emplace(f->get_children().size());
		}
	}
	if (branches.empty())
	{
		// for some reason...
		return;
	}
	rules.push_back(opt::OptRule{
		[branches](query::Query& q)
		{
			q = q.select("root");
			for (auto& branch : branches)
			{
				for (size_t bfactor : branch.second)
				{
					q = q.where(get_cstsource(branch.first, bfactor));
				}
			}
		}, std::make_shared<ConstantTarget<T>>(graph)});
}

}

#endif // ETEQ_OPT_CSTRULES_HPP
