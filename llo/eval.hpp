#include "ade/traveler.hpp"

#include "llo/opmap.hpp"
#include "llo/data.hpp"

#ifndef LLO_EVAL_HPP
#define LLO_EVAL_HPP

namespace llo
{

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
		const char* data = leaf->data();
		DTYPE dtype = (DTYPE) leaf->type_code();
		const ade::Shape& shape = leaf->shape();
		out_ = GenericData(shape, dtype_);
		out_.copyover(data, dtype);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		age::_GENERATED_OPCODES opcode = (age::_GENERATED_OPCODES)
			func->get_opcode().code_;
		out_ = GenericData(func->shape(), dtype_);

		ade::ArgsT children = func->get_children();
		uint8_t nargs = children.size();
		DataArgsT argdata = DataArgsT(nargs);
		if (func->get_opcode().code_ == age::RAND_BINO)
		{
			if (nargs != 2)
			{
				err::fatalf("cannot RAND_BINO without exactly 2 arguments: "
					"using %d arguments", nargs);
			}
			Evaluator left_eval(dtype_);
			children[0].tensor_->accept(left_eval);
			argdata[0] = {children[0].mapper_, left_eval.out_};

			Evaluator right_eval(DOUBLE);
			children[1].tensor_->accept(right_eval);
			argdata[1] = {children[0].mapper_, right_eval.out_};
		}
		else
		{
			for (uint8_t i = 0; i < nargs; ++i)
			{
				Evaluator evaler(dtype_);
				children[i].tensor_->accept(evaler);
				argdata[i] = {children[i].mapper_, evaler.out_};
			}
		}

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
