
#ifndef EXPERIMENTAL_OPT_TARGET_HPP
#define EXPERIMENTAL_OPT_TARGET_HPP

#include "teq/logs.hpp"

#include "query/query.hpp"

#include "experimental/opt/optimize.pb.h"

namespace opt
{

struct TargetResult final
{
	union Result
	{
		teq::TensptrT tens_;
		double scalar_;
	} result_;
	bool is_tens_;
};

struct iTarget
{
	virtual ~iTarget (void) = default;

	virtual TargetResult convert (const teq::Shape& outshape,
		const query::SymbMapT& candidates) const = 0;
};

using TargptrT = std::shared_ptr<iTarget>;

using TargptrsT = std::vector<TargptrT>;

struct iTargetFactory
{
	virtual ~iTargetFactory (void) = default;

	virtual TargptrT make_scalar (double scalar) const = 0;

	virtual TargptrT make_symbol (std::string symbol) const = 0;

	virtual TargptrT make_functor (std::string opname,
		const google::protobuf::Map<std::string,query::Attribute>& attrs,
		const TargptrsT& args) const = 0;
};

TargptrT parse_target (const query::Node& root,
	const iTargetFactory& builder)
{
	TargptrT out;
	switch (root.val_case())
	{
		case query::Node::ValCase::kCst:
			out = builder.make_scalar(root.cst());
			break;
		case query::Node::ValCase::kSymb:
			out = builder.make_symbol(root.symb());
			break;
		case query::Node::ValCase::kOp:
		{
			const query::Operator& pb_op = root.op();
			const auto& pb_args = pb_op.args();
			TargptrsT args;
			args.reserve(pb_args.size());
			std::transform(pb_args.begin(), pb_args.end(),
				std::back_inserter(args),
				[&](const query::Node& pb_arg)
				{
					return parse_target(pb_arg, builder);
				});
			out = builder.make_functor(pb_op.opname(),
				pb_op.attrs(), args);
		}
			break;
		case query::Node::ValCase::kVar:
			teq::fatal("cannot specify variable in dest subgraph");
			break;
		default:
			teq::fatal("cannot parse unknown element");
	}
	return out;
}

}

#endif // EXPERIMENTAL_OPT_TARGET_HPP
