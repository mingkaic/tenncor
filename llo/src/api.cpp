#include "ade/functor.hpp"

#include "llo/api.hpp"

#ifdef LLO_API_HPP

namespace llo
{

DataNode one (void)
{
	return DataNode(EvalCtx(), ade::Tensor::SYMBOLIC_ONE);
}

DataNode abs (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::ABS>::get({arg.tensor_}));
}

DataNode neg (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::NEG>::get({arg.tensor_}));
}

DataNode bit_not (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::NOT>::get({arg.tensor_}));
}

DataNode sin (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::SIN>::get({arg.tensor_}));
}

DataNode cos (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::COS>::get({arg.tensor_}));
}

DataNode tan (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::TAN>::get({arg.tensor_}));
}

DataNode exp (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::EXP>::get({arg.tensor_}));
}

DataNode log (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::LOG>::get({arg.tensor_}));
}

DataNode sqrt (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::SQRT>::get({arg.tensor_}));
}

DataNode round (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::ROUND>::get({arg.tensor_}));
}

DataNode flip (DataNode arg, uint8_t dim)
{
	return FuncWrapper<uint8_t>::get(arg.ctx_, std::shared_ptr<ade::iFunctor>(
		ade::Functor<ade::FLIP>::get({arg.tensor_})), dim);
}

DataNode pow (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::POW>::get({a.tensor_, b.tensor_}));
}

DataNode add (DataNode a, DataNode b)
{
	return sum({a, b});
}

DataNode sum (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	std::vector<ade::Tensorptr> tens;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		tens.push_back(arg.tensor_);
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::ADD>::get(tens));
}

DataNode sub (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::SUB>::get({a.tensor_, b.tensor_}));
}

DataNode mul (DataNode a, DataNode b)
{
	return prod({a, b});
}

DataNode prod (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	std::vector<ade::Tensorptr> tens;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		tens.push_back(arg.tensor_);
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::MUL>::get(tens));
}

DataNode div (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::DIV>::get({a.tensor_, b.tensor_}));
}

DataNode eq (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::EQ>::get({a.tensor_, b.tensor_}));
}

DataNode neq (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::NE>::get({a.tensor_, b.tensor_}));
}

DataNode lt (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::LT>::get({a.tensor_, b.tensor_}));
}

DataNode gt (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::GT>::get({a.tensor_, b.tensor_}));
}

DataNode min (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	std::vector<ade::Tensorptr> tens;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		tens.push_back(arg.tensor_);
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::MIN>::get(tens));
}

DataNode max (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	std::vector<ade::Tensorptr> tens;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		tens.push_back(arg.tensor_);
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::MAX>::get(tens));
}

DataNode clip (DataNode x, DataNode lo, DataNode hi)
{
	return min({max({x, lo}), hi});
}

DataNode rand_binom (DataNode ntrials, DataNode prob)
{
	return DataNode(EvalCtx({&ntrials.ctx_, &prob.ctx_}),
		ade::Functor<ade::RAND_BINO>::get({ntrials.tensor_, prob.tensor_}));
}

DataNode rand_uniform (DataNode lower, DataNode upper)
{
	return DataNode(EvalCtx({&lower.ctx_, &upper.ctx_}),
		ade::Functor<ade::RAND_UNIF>::get({lower.tensor_, upper.tensor_}));
}

DataNode rand_normal (DataNode mean, DataNode stdev)
{
	return DataNode(EvalCtx({&mean.ctx_, &stdev.ctx_}),
		ade::Functor<ade::RAND_NORM>::get({mean.tensor_, stdev.tensor_}));
}

DataNode n_elems (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::N_ELEMS>::get({arg.tensor_}));
}

DataNode n_dims (DataNode arg, uint8_t dim)
{
	return FuncWrapper<uint8_t>::get(arg.ctx_, std::shared_ptr<ade::iFunctor>(
		ade::Functor<ade::N_DIMS>::get({arg.tensor_})), dim);
}

DataNode argmax (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::ARGMAX,uint8_t>::get(
		{arg.tensor_}, 8));
}

DataNode reduce_max (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::RMAX,uint8_t>::get(
		{arg.tensor_}, 8));
}

DataNode reduce_sum (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::RSUM,uint8_t>::get(
		{arg.tensor_}, 8));
}

DataNode reduce_sum (DataNode arg, uint8_t groupidx)
{
	return DataNode(arg.ctx_, ade::Functor<ade::RSUM,uint8_t>::get(
		{arg.tensor_}, groupidx));
}

DataNode matmul (DataNode a, DataNode b)
{
	return matmul(a, b, 1, 1);
}

DataNode matmul (DataNode a, DataNode b,
	uint8_t agroup_idx, uint8_t bgroup_idx)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}), ade::Functor<ade::MATMUL,
		uint8_t,uint8_t>::get({a.tensor_, b.tensor_}, agroup_idx, bgroup_idx));
}

DataNode convolute (DataNode canvas, DataNode window)
{
	throw std::bad_function_call(); // unimplemented
}

DataNode permute (DataNode arg, std::vector<uint8_t> order)
{
	return DataNode(arg.ctx_, ade::Functor<ade::PERMUTE,
		std::vector<uint8_t>>::get({arg.tensor_}, order));
}

DataNode extend (DataNode arg, std::vector<uint8_t> ext)
{
	return DataNode(arg.ctx_, ade::Functor<ade::EXTEND,
		std::vector<ade::DimT>>::get({arg.tensor_}, ext));
}

DataNode reshape (DataNode arg, std::vector<uint8_t> slist)
{
	return DataNode(arg.ctx_, ade::Functor<ade::RESHAPE,
		std::vector<ade::DimT>>::get({arg.tensor_}, slist));
}

}

#endif
