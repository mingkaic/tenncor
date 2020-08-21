#include <google/protobuf/util/json_util.h>

#include "internal/opt/optimize.pb.h"

#include "internal/opt/parse.hpp"

#ifdef OPT_PARSE_HPP

namespace opt
{

TargptrT parse_target (const TargetNode& root,
	const iTargetFactory& builder)
{
	TargptrT out;
	switch (root.val_case())
	{
		case TargetNode::ValCase::kCst:
		{
			const Scalar& cst = root.cst();
			out = builder.make_scalar(cst.value(), cst.shape());
		}
			break;
		case TargetNode::ValCase::kSymb:
			out = builder.make_symbol(root.symb());
			break;
		case TargetNode::ValCase::kOp:
		{
			const TargOp& pb_op = root.op();
			const auto& pb_args = pb_op.args();
			TargptrsT args;
			args.reserve(pb_args.size());
			std::transform(pb_args.begin(), pb_args.end(),
				std::back_inserter(args),
				[&](const TargetNode& pb_arg)
				{
					return parse_target(pb_arg, builder);
				});
			out = builder.make_functor(pb_op.opname(),
				pb_op.attrs(), args);
		}
			break;
		default:
			global::fatal("cannot parse unknown element");
	}
	return out;
}

void parse_optimization (OptRulesT& rules,
	const Optimization& pb_opt, const iTargetFactory& tfactory)
{
	for (const Conversion& pb_conv : pb_opt.conversions())
	{
		const auto& pb_srcs = pb_conv.srcs();
		const TargetNode& pb_dest = pb_conv.dest();
		TargptrT target = parse_target(pb_dest, tfactory);
		rules.push_back(OptRule{pb_srcs, target});
	}
}

void json_parse (OptRulesT& rules,
	std::istream& json_in, const iTargetFactory& tfactory)
{
	std::string jstr(std::istreambuf_iterator<char>(json_in), {});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;
	Optimization optimization;
	if (google::protobuf::util::Status::OK !=
		google::protobuf::util::JsonStringToMessage(
			jstr, &optimization, options))
	{
		global::fatal("failed to parse json optimization");
	}
	parse_optimization(rules, optimization, tfactory);
}

marsh::iObject* parse (const query::Attribute& pba, const opt::GraphInfo& graphinfo)
{
	marsh::iObject* out = nullptr;
	switch (pba.attr_case())
	{
		case query::Attribute::kInum:
			out = new marsh::Number<int64_t>(pba.inum());
			break;
		case query::Attribute::kDnum:
			out = new marsh::Number<double>(pba.dnum());
			break;
		case query::Attribute::kIarr:
		{
			const auto& arr = pba.iarr().values();
			out = new marsh::NumArray<int64_t>(
				std::vector<int64_t>(arr.begin(), arr.end()));
		}
			break;
		case query::Attribute::kDarr:
		{
			const auto& arr = pba.darr().values();
			out = new marsh::NumArray<double>(
				std::vector<double>(arr.begin(), arr.end()));
		}
			break;
		case query::Attribute::kStr:
			out = new marsh::String(pba.str());
			break;
		case query::Attribute::kNode:
		{
			auto results = graphinfo.find(pba.node());
			if (results.size() > 0)
			{
				global::fatal("ambiguous node attribute");
			}
			out = new teq::TensorObj(results.front());
		}
			break;
		case query::Attribute::kLayer:
		{
			const query::Layer& layer = pba.layer();
			if (false == layer.has_input() ||
				query::Layer::kNameNil == layer.nullable_name_case())
			{
				global::fatal("cannot parse layer attribute unnamed or without input");
			}
			auto results = graphinfo.find(layer.input());
			if (results.size() > 0)
			{
				global::fatal("ambiguous layer attribute");
			}
			out = new teq::LayerObj(layer.name(), results.front());
		}
			break;
		default:
			global::fatal("cannot parse unknown attribute");
	}
	return out;
}

}

#endif
