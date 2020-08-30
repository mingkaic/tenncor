
#include "internal/teq/evaluator.hpp"

#ifdef TEQ_EVALUATOR_HPP

namespace teq
{

const std::string eval_key = "evaluater";

void set_eval (iEvaluator* eval, global::CfgMapptrT ctx)
{
	ctx->rm_entry(eval_key);
	if (eval)
	{
		ctx->template add_entry<iEvaluator>(eval_key,
			[=](){ return eval; });
	}
}

iEvaluator& get_eval (const global::CfgMapptrT& ctx)
{
	auto eval = static_cast<iEvaluator*>(ctx->get_obj(eval_key));
	if (nullptr == eval)
	{
		eval = new Evaluator();
		set_eval(eval, ctx);
	}
	return *eval;
}

}

#endif
