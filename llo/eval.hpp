#include "ade/traveler.hpp"

#include "llo/opmap.hpp"
#include "llo/data.hpp"

#ifndef LLO_EVAL_HPP
#define LLO_EVAL_HPP

namespace llo
{

void calc_func_args (DataArgsT& out, DTYPE dtype, ade::iFunctor* func);

/// Visitor implementation to evaluate ade nodes according to ctx and dtype
/// Given a global context containing ade-llo association maps, get data from
/// llo::Sources when possible, otherwise treat native ade::Tensors as zeroes
/// Additionally, Evaluator attempts to get meta-data from llo::FuncWrapper
/// before checking native ade::Functor
struct Evaluator final : public ade::iTraveler
{
    Evaluator (DTYPE dtype) : dtype_(dtype) {}

	/// Implementation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
        ade::iData& data = leaf->data();
		out_ = GenericData(leaf->shape(), dtype_, "unlabelled");
		out_.take_astype(dtype_, data);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		age::_GENERATED_OPCODES opcode = (age::_GENERATED_OPCODES)
            func->get_opcode().code_;
		out_ = GenericData(func->shape(), dtype_, "unlabelled");

		DataArgsT argdata;
		calc_func_args(argdata, dtype_, func);
		op_exec(opcode, out_, argdata);
	}

	/// Output data evaluated upon visiting node
	GenericData out_;

private:
	/// Output type when evaluating data
	DTYPE dtype_;
};

}

#endif // LLO_EVAL_HPP
