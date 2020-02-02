
#ifndef EXPERIMENTAL_OPT_PARSE_HPP
#define EXPERIMENTAL_OPT_PARSE_HPP

#include <google/protobuf/util/json_util.h>

#include "teq/logs.hpp"

#include "experimental/opt/optimize.pb.h"

namespace opt
{

struct OpInfo final
{
	std::string opname_;

	std::vector<size_t> args_;

	std::unordered_map<std::string,Attribute> attrs_;
};

using AnyMapT = std::unordered_map<std::string,teq::TensptrT>;

struct OConversion final
{
	void build (size_t typecode, const AnyMapT& anys) const
	{
		std::vector<std::string> keys;
		keys.reserve(anys.size());
		for (const auto& a : anys)
		{
			keys.push_back(a.first);
		}
		std::sort(keys.begin(), keys.end());
		teq::TensptrsT nodes;
		nodes.reserve(keys.size());
		std::transform(keys.begin(), keys.end(),
			std::back_inserter(nodes),
			[&](std::string key) { return anys.at(key); });
		for (const OpInfo& op : ops_)
		{
			teq::TensptrsT args;
			args.reserve(op.args_);
			std::transform(op.args_.begin(), op.args_.end(),
				std::back_inserter(args),
				[](size_t idx) { return nodes.at(idx); });
			nodes.push_Back(builder_(op.opname_, op.attrs_, args));
		}
	}

	std::vector<OpInfo> ops_;

	iBuilder& builder_;
};

using OConvptrT = std::shared_ptr<OConversion>;

using OConValT = std::vector<std::pair<std::string,OConvptrT>;

using OConvTrieT = estd::Trie<query::PathNodesT,
	estd::TrieBigNode<query::PathNode,OConValT,query::search::PathNodeHasher>>;

struct iOptNode
{
	//
};

using ONodeptrT = std::shared_ptr<iOptNode>;

using ONodeptrsT = std::vector<ONodeptrT>;

// Abstract factory for easy parsing
struct iOptimizeFactory
{
	virtual ~iOptimizeFactory (void) = default;

	virtual ONodeptrT make_functor (std::string opname,
		const google::protobuf::Map<std::string,Attribute>& attrs,
		const ONodeptrsT& args) const = 0;

	virtual ONodeptrT make_scalar (double scalar) const = 0;

	virtual ONodeptrT make_variable (
		std::optional<std::string> label,
		std::optional<std::string> dtype,
		const std::vector<uint32_t>& shape) const = 0;

	virtual ONodeptrT make_any (std::string symbol) const = 0;

	virtual ONodeptrT make_variadic (std::string symbol) const = 0;
};

struct iMatcher : public iOptNode
{
	//
};

using MatchptrT = std::shared_ptr<iMatcher>;

using MatchptrsT = std::vector<MatchptrT>;

struct FuncMatcher final : public iMatcher
{
	FuncMatcher (const std::string& opname,
		const google::protobuf::Map<std::string,Attribute>& attrs,
		const MatchptrsT& margs) : opname_(opname), margs_(margs)
	{
		//
	}

	void match (OConvTrieT::NodeT* tnode)
	{
		//
	}

	std::string opname_;

	MatchptrsT margs_;
};

struct CstMatcher final : public iMatcher
{
	CstMatcher (double scalar) : scalar_(scalar) {}

	void match (OConvTrieT::NodeT* tnode)
	{
		//
	}

	double scalar_;
};

struct VareMatcher final : public iMatcher
{
	CstMatcher (double scalar) : scalar_(scalar) {}

	void match (OConvTrieT::NodeT* tnode)
	{
		//
	}

	double scalar_;
}

struct MatcherFactory final : public iOptimizeFactory
{
	ONodeptrT make_functor (std::string opname,
		const google::protobuf::Map<std::string,Attribute>& attrs,
		const ONodeptrsT& args) const override
	{
		MatchptrsT margs;
		margs.reserve(args.size());
		std::transform(args.begin(), args.end(), std::back_inserter(margs),
			[](ONodeptrT arg)
			{ return std::static_pointer_cast<iMatcher>(arg); });
		return std::make_shared<FuncMatcher>(opname, attrs, margs);
	}

	ONodeptrT make_scalar (double scalar) const override
	{
		return std::make_shared<CstMatcher>(scalar);
	}

	ONodeptrT make_variable (
		std::optional<std::string> label,
		std::optional<std::string> dtype,
		const std::vector<uint32_t>& shape) const override
	{
		//
	}

	ONodeptrT make_any (std::string symbol) const override
	{
		//
	}

	ONodeptrT make_variadic (std::string symbol) const override
	{
		//
	}
};

struct iTarget : public iOptNode
{
	virtual ~iTarget (void) = default;

	virtual teq::TensptrT convert (teq::Shape outshape,
		const Candidate& candidate) const = 0;
};

using TargptrT = std::shared_ptr<iTarget>;

using TargptrsT = std::vector<TargptrT>;

const std::string var_prefix = "var_";
const std::string any_prefix = "any_";
const std::string variadic_prefix = "..._";

struct iBuilderFactory : public iOptimizeFactory
{
	virtual ~iBuilderFactory (void) = default;

	virtual TargptrT make_symbol (std::string symbol) const = 0;

	ONodeptrT make_variable (
		std::optional<std::string> label,
		std::optional<std::string> dtype,
		const std::vector<uint32_t>& shape) const override
	{
		if (false == label.has_value())
		{
			teq::fatal("cannot make builder of ambiguous variable");
		}
		return make_symbol(var_prefix + *label);
	}

	ONodeptrT make_any (std::string symbol) const override
	{
		return make_symbol(any_prefix + symbol);
	}

	ONodeptrT make_variadic (std::string symbol) const override
	{
		return make_symbol(variadic_prefix + symbol);
	}
};

ONodeptrT parse_node (const Node& root, const iOptimizeFactory& builder)
{
	ONodeptrT out;
	switch (root.val_case())
	{
		case Node::ValCase::kCst:
			out = builder.make_scalar(root.cst());
			break;
		case Node::ValCase::kVar:
		{
			const auto& pb_var = root.var();
			const auto& pb_shape = pb_var.shape();
			std::optional<std::string> label;
			std::optional<std::string> dtype;
			if (Variable::kLabel == pb_var.nullable_label_case())
			{
				label = pb_var.label();
			}
			if (Variable::kDtype == pb_var.nullable_dtype_case())
			{
				dtype = pb_var.dtype();
			}
			out = builder.make_variable(label, dtype,
				std::vector<uint32_t>(pb_shape.begin(), pb_shape.end()));
		}
			break;
		case Node::ValCase::kOp:
		{
			const Operator& pb_op = root.op();
			const auto& pb_args = pb_op.args();
			std::vector<T> args;
			args.reserve(pb_args.size());
			std::transform(pb_args.begin(), pb_args.end(),
				std::back_inserter(args),
				[&builder](const Node& arg) { return parse_node(arg, builder); });
			out = builder.make_functor(pb_op.opname(), pb_op.attrs(), args);
		}
			break;
		case Node::ValCase::kAny:
			out = builder.make_any(root.any());
			break;
		case Node::ValCase::kVariadic:
			out = builder.make_variadic(root.variadic());
			break;
		default:
			teq::fatal("cannot parse unknown element");
	}
	return out;
}

void parse_optimization (OConvTrieT& trie, const Optimization& pb_opt, const iBuilderFactory& bfactory)
{
	MatcherFactory mfactory;
	for (const Conversion& pb_conv : pb_opt.conversions())
	{
		const query::Node& pb_dest = pb_conv.dest();
		auto conv = std::static_pointer_cast<iTarget>(
			parse_node(pb_dest, bfactory));
		for (const query::Node& pb_src : pb_conv.srcs())
		{
			// [&pb_src](teq::TensSet& results, query::Query& q)
			// {
			//     q.where(pb_src).exec(results);
			// };
		}
	}
}

void json_parse (OConvTrieT& trie, std::istream& json_in)
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
	parse_optimization (trie, optimization);
}

}

#endif // EXPERIMENTAL_OPT_PARSE_HPP
