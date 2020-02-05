
#ifndef EXPERIMENTAL_OPT_PARSE_HPP
#define EXPERIMENTAL_OPT_PARSE_HPP

#include <google/protobuf/util/json_util.h>

#include "experimental/opt/target.hpp"
#include "experimental/opt/rule.hpp"

namespace opt
{

void find_symbols (std::unordered_set<std::string>& out,
	const query::Node& cond)
{
	switch (cond.val_case())
	{
		case query::Node::ValCase::kSymb:
			out.emplace(cond.symb());
			break;
		case query::Node::ValCase::kOp:
			for (const query::Node& arg : cond.op().args())
			{
				find_symbols(out, arg);
			}
			break;
		default:
			break;
	}
}

void parse_optimization (OptRulesT& rules,
	const Optimization& pb_opt, const iTargetFactory& tfactory)
{
	for (const Conversion& pb_conv : pb_opt.conversions())
	{
		std::unordered_set<std::string> relevant_anys;
		const auto& pb_srcs = pb_conv.srcs();
		const query::Node& pb_dest = pb_conv.dest();
		for (const query::Node& pb_src : pb_srcs)
		{
			find_symbols(relevant_anys, pb_src);
		}
		MatcherF matcher = [pb_srcs, relevant_anys](query::Query& q)
		{
			for (std::string any : relevant_anys)
			{
				q = q.select(any);
			}
			for (const query::Node& pb_src : pb_srcs)
			{
				q = q.where(std::make_shared<query::Node>(pb_src));
			}
		};
		TargptrT target = parse_target(pb_dest, tfactory);
		rules.push_back(OptRule{matcher, target});
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
		teq::fatal("failed to parse json optimization");
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
				teq::fatal("ambiguous node attribute");
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
				teq::fatal("cannot parse layer attribute unnamed or without input");
			}
			auto results = graphinfo.find(layer.input());
			if (results.size() > 0)
			{
				teq::fatal("ambiguous layer attribute");
			}
			out = new teq::LayerObj(layer.name(), results.front());
		}
			break;
		default:
			teq::fatal("cannot parse unknown attribute");
	}
	return out;
}

}

#endif // EXPERIMENTAL_OPT_PARSE_HPP
