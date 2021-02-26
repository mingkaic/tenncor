
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

#ifndef HONE_CSTRULES_HPP
#define HONE_CSTRULES_HPP

#include "internal/opt/opt.hpp"

#include "tenncor/eteq/eteq.hpp"

namespace hone
{

#define _CHOOSE_CST_TARGETTYPE(REALTYPE)\
if (auto sinfo = eigen::sparse_info(*root)){\
	out = eteq::make_constant_tensor<REALTYPE>((REALTYPE*)data, *sinfo, root->shape());\
}else{\
	out = eteq::make_constant_tensor<REALTYPE>((REALTYPE*)data, root->shape());\
}

// custom target for calculating constant values
struct ConstantTarget final : public opt::iTarget
{
	ConstantTarget (const opt::GraphInfo& graph,
		global::CfgMapptrT context = global::context()) : graph_(&graph), ctx_(context) {}

	teq::TensptrT convert (const query::SymbMapT& candidates) const override
	{
		teq::iTensor* root = candidates.at("root");
		eigen::Device device;
		teq::get_eval(ctx_).evaluate(device, {root});
		void* data = root->device().data();
		auto outtype = (egen::_GENERATED_DTYPE) root->get_meta().type_code();
		teq::TensptrT out;
		TYPE_LOOKUP(_CHOOSE_CST_TARGETTYPE, outtype);
		return out;
	}

	const opt::GraphInfo* graph_;

	global::CfgMapptrT ctx_;
};

#undef _CHOOSE_CST_TARGETTYPE

// source graph for certain branching factor of certain operator
static inline void get_cstsource (query::Node& node, std::string opname, size_t nbranch)
{
	query::Operator* op = node.mutable_op();
	op->set_opname(opname);
	op->set_capture("root");
	for (size_t i = 0; i < nbranch; ++i)
	{
		query::Node* arg = op->add_args();
		query::Leaf* leaf = arg->mutable_leaf();
		leaf->set_usage(teq::get_usage_name(teq::Usage::IMMUTABLE));
	}
}

// gather branching factor of all operators,
// then append rule converting constant source to constant target
void generate_cstrules (opt::OptRulesT& rules,
	const opt::GraphInfo& graph,
	global::CfgMapptrT ctx = global::context());

}

#endif // HONE_CSTRULES_HPP
