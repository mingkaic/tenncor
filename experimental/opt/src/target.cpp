#include "experimental/opt/target.hpp"

#ifdef EXPERIMENTAL_OPT_TARGET_HPP

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
			teq::fatal("cannot parse unknown element");
	}
	return out;
}

}

#endif
