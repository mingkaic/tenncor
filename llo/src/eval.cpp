#include "llo/eval.hpp"

#ifdef LLO_EVAL_HPP

namespace llo
{

#define FILL_ONE(TYPE){ TYPE* ptr = (TYPE*) cptr;\
std::fill(ptr, ptr + n, (TYPE) 1); } break;

// fill all elements of specified type under cptr with values of 1
static void fill_one (char* cptr, size_t n, DTYPE dtype)
{
	switch (dtype)
	{
		case DOUBLE:
			FILL_ONE(double)
		case FLOAT:
			FILL_ONE(float)
		case INT8:
			FILL_ONE(int8_t)
		case INT16:
			FILL_ONE(int16_t)
		case INT32:
			FILL_ONE(int32_t)
		case INT64:
			FILL_ONE(int64_t)
		case UINT8:
			FILL_ONE(uint8_t)
		case UINT16:
			FILL_ONE(uint16_t)
		case UINT32:
			FILL_ONE(uint32_t)
		case UINT64:
			FILL_ONE(uint64_t)
		default:
			ade::fatal("evaluating unknown type");
	}
}

void Evaluator::visit (ade::Tensor* leaf)
{
	if (leaf == ade::Tensor::SYMBOLIC_ONE.get())
	{
		out_ = GenericData(ade::Shape(), dtype_);
		fill_one(out_.data_.get(), 1, dtype_);
		return;
	}
	auto srcpair = ctx_->srcs_.find(leaf);
	if (ctx_->srcs_.end() != srcpair)
	{
		out_ = srcpair->second->data(dtype_);
	}
	else
	{
		if (leaf != ade::Tensor::SYMBOLIC_ZERO.get())
		{
			ade::warnf("evaluating an ade::Tensor %s without associated "
				"Source according to input context... treating data as 0",
				leaf->to_string().c_str()); // todo: describe ctx for comprehensive report
		}
		out_ = GenericData(ade::Shape(), dtype_);
		std::memset(out_.data_.get(), 0, type_size(dtype_));
	}
}

void Evaluator::visit (ade::iFunctor* func)
{
	auto funkpair = ctx_->funks_.find(func);
	if (ctx_->funks_.end() != funkpair)
	{
		out_ = funkpair->second->data(dtype_);
		return;
	}
	// else visit pure ade::iFunctor
	ade::OPCODE opcode = func->get_code();

	out_ = GenericData(func->shape(), dtype_);

	std::vector<GenericData> argdata;
	get_func_children(argdata, *ctx_, dtype_, func);
	// todo: think of better way to deal with these metadata
	// (perhaps pass around as metadata interface of template implementation)
	switch (opcode)
	{
		case ade::ARGMAX:
		{
			auto mf = static_cast<ade::Functor<ade::ARGMAX,uint8_t>*>(func);
			op_exec(opcode, out_, argdata, std::get<0>(mf->meta()));
		}
		break;
		case ade::RSUM:
		{
			auto mf = static_cast<ade::Functor<ade::RSUM,uint8_t>*>(func);
			op_exec(opcode, out_, argdata, std::get<0>(mf->meta()));
		}
		break;
		case ade::RMAX:
		{
			auto mf = static_cast<ade::Functor<ade::RMAX,uint8_t>*>(func);
			op_exec(opcode, out_, argdata, std::get<0>(mf->meta()));
		}
		break;
		case ade::MATMUL:
		{
			auto mf = static_cast<ade::Functor<
				ade::MATMUL,uint8_t,uint8_t>*>(func);
			op_exec(opcode, out_, argdata,
				std::get<0>(mf->meta()), std::get<1>(mf->meta()));
		}
		break;
		case ade::PERMUTE:
		{
			auto pf = static_cast<ade::Functor<
				ade::PERMUTE,std::vector<uint8_t>>*>(func);
			op_exec(opcode, out_, argdata, std::get<0>(pf->meta()));
		}
		break;
		case ade::EXTEND:
		{
			auto ef = static_cast<ade::Functor<
				ade::EXTEND,std::vector<ade::DimT>>*>(func);
			op_exec(opcode, out_, argdata, std::get<0>(ef->meta()));
		}
		break;
		default:
			op_exec(opcode, out_, argdata);
	}
}

void get_func_children (std::vector<GenericData>& out,
	const EvalCtx& ctx, DTYPE dtype, ade::iFunctor* func)
{
	std::vector<ade::iTensor*> children = func->get_children();
	uint8_t nargs = children.size();
	out = std::vector<GenericData>(nargs);
	if (func->get_code() == ade::RAND_BINO)
	{
		if (nargs != 2)
		{
			ade::fatalf("cannot RAND_BINO without 2 arguments: "
				"using %d arguments", nargs);
		}
		Evaluator left_eval(ctx, dtype);
		children[0]->accept(left_eval);
		out[0] = left_eval.out_;

		Evaluator right_eval(ctx, DOUBLE);
		children[1]->accept(right_eval);
		out[1] = right_eval.out_;
	}
	else
	{
		for (uint8_t i = 0; i < nargs; ++i)
		{
			Evaluator evaler(ctx, dtype);
			children[i]->accept(evaler);
			out[i] = evaler.out_;
		}
	}
}

#undef FILL_ONE

}

#endif
