#include "llo/eval.hpp"

#ifdef LLO_EVAL_HPP

namespace llo
{

void calc_func_args (DataArgsT& out, const EvalCtx& ctx,
	DTYPE dtype, ade::iFunctor* func)
{
	ade::ArgsT children = func->get_children();
	uint8_t nargs = children.size();
	out = DataArgsT(nargs);
	if (func->get_code().opnum() == age::RAND_BINO)
	{
		if (nargs != 2)
		{
			err::fatalf("cannot RAND_BINO without exactly 2 arguments: "
				"using %d arguments", nargs);
		}
		Evaluator left_eval(ctx, dtype);
		children[0].tensor_->accept(left_eval);
		out[0] = {children[0].mapper_, left_eval.out_};

		Evaluator right_eval(ctx, DOUBLE);
		children[1].tensor_->accept(right_eval);
		out[1] = {children[0].mapper_, right_eval.out_};
	}
	else
	{
		for (uint8_t i = 0; i < nargs; ++i)
		{
			Evaluator evaler(ctx, dtype);
			children[i].tensor_->accept(evaler);
			out[i] = {children[i].mapper_, evaler.out_};
		}
	}
}

}

#endif
