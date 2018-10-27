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
	return DataNode(arg.ctx_, ade::Functor<ade::ABS>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode neg (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::NEG>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode bit_not (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::NOT>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode sin (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::SIN>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode cos (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::COS>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode tan (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::TAN>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode exp (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::EXP>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode log (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::LOG>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode sqrt (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::SQRT>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode round (DataNode arg)
{
	return DataNode(arg.ctx_, ade::Functor<ade::ROUND>::get({
		{ade::identity, arg.tensor_}}));
}

DataNode flip (DataNode arg, uint8_t dim)
{
	return DataNode(arg.ctx_, ade::Functor<ade::COPY>::get({
		{ade::flip(dim), arg.tensor_}}));
}

DataNode pow (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::POW>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode add (DataNode a, DataNode b)
{
	return sum({a, b});
}

DataNode sum (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	ade::ArgsT ade_args;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		ade_args.push_back({ade::identity, arg.tensor_});
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::ADD>::get(ade_args));
}

DataNode sub (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::SUB>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode mul (DataNode a, DataNode b)
{
	return prod({a, b});
}

DataNode prod (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	ade::ArgsT ade_args;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		ade_args.push_back({ade::identity, arg.tensor_});
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::MUL>::get(ade_args));
}

DataNode div (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::DIV>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode eq (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::EQ>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode neq (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::NE>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode lt (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::LT>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode gt (DataNode a, DataNode b)
{
	return DataNode(EvalCtx({&a.ctx_, &b.ctx_}),
		ade::Functor<ade::GT>::get({
			{ade::identity, a.tensor_},
			{ade::identity, b.tensor_}
		}));
}

DataNode min (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	ade::ArgsT ade_args;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		ade_args.push_back({ade::identity, arg.tensor_});
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::MIN>::get(ade_args));
}

DataNode max (std::vector<DataNode> args)
{
	std::vector<const EvalCtx*> contexas;
	ade::ArgsT ade_args;
	for (DataNode& arg : args)
	{
		contexas.push_back(&arg.ctx_);
		ade_args.push_back({ade::identity, arg.tensor_});
	}
	return DataNode(EvalCtx(contexas), ade::Functor<ade::MAX>::get(ade_args));
}

DataNode clip (DataNode x, DataNode lo, DataNode hi)
{
	return min({max({x, lo}), hi});
}

DataNode rand_binom (DataNode ntrials, DataNode prob)
{
	return DataNode(EvalCtx({&ntrials.ctx_, &prob.ctx_}),
		ade::Functor<ade::RAND_BINO>::get({
			{ade::identity, ntrials.tensor_},
			{ade::identity, prob.tensor_}
		}));
}

DataNode rand_uniform (DataNode lower, DataNode upper)
{
	return DataNode(EvalCtx({&lower.ctx_, &upper.ctx_}),
		ade::Functor<ade::RAND_UNIF>::get({
			{ade::identity, lower.tensor_},
			{ade::identity, upper.tensor_}
		}));
}

DataNode rand_normal (DataNode mean, DataNode stdev)
{
	return DataNode(EvalCtx({&mean.ctx_, &stdev.ctx_}),
		ade::Functor<ade::RAND_NORM>::get({
			{ade::identity, mean.tensor_},
			{ade::identity, stdev.tensor_}
		}));
}

DataNode n_elems (DataNode arg)
{
	return Source<uint64_t>::get_scalar(arg.tensor_->shape().n_elems());
}

DataNode n_dims (DataNode arg, uint8_t dim)
{
	return Source<uint8_t>::get_scalar(arg.tensor_->shape().at(dim));
}

DataNode reduce_max (DataNode arg)
{
	return reduce_max(arg, 0);
}

DataNode reduce_max (DataNode arg, uint8_t groupidx)
{
	const ade::Shape& shape = arg.tensor_->shape();
	std::vector<ade::DimT> slist(shape.begin() + groupidx, shape.end());
	return DataNode(arg.ctx_, ade::Functor<ade::MAX>::get(
		{{ade::reduce(groupidx, slist), arg.tensor_}}));
}

DataNode reduce_sum (DataNode arg)
{
	return reduce_sum(arg, 0);
}

DataNode reduce_sum (DataNode arg, uint8_t groupidx)
{
	const ade::Shape& shape = arg.tensor_->shape();
	std::vector<ade::DimT> slist(shape.begin() + groupidx, shape.end());
	return DataNode(arg.ctx_, ade::Functor<ade::ADD>::get({
		{ade::reduce(groupidx, slist), arg.tensor_}}));
}

DataNode permute (DataNode arg, std::vector<uint8_t> order)
{
	return DataNode(arg.ctx_, ade::Functor<ade::COPY>::get({
		{ade::permute(order), arg.tensor_}}));
}

DataNode extend (DataNode arg, uint8_t after, std::vector<uint8_t> ext)
{
	return DataNode(arg.ctx_, ade::Functor<ade::COPY>::get({
		{ade::extend(after, ext), arg.tensor_}}));
}

DataNode matmul (DataNode a, DataNode b)
{
	const ade::Shape& ashape = a.tensor_->shape();
	const ade::Shape& bshape = b.tensor_->shape();
	if (std::any_of(ashape.begin() + 2, ashape.end(),
		[](ade::DimT d) { return 1 != d; }) ||
		std::any_of(bshape.begin() + 2, bshape.end(),
		[](ade::DimT d) { return 1 != d; }))
	{
		ade::fatalf("cannot matmul with non-2D shapes %s and %s",
			ashape.to_string().c_str(), bshape.to_string().c_str());
	}
	if (ashape.at(0) != bshape.at(1))
	{
		ade::fatalf("cannot matmul with incompatible common dimensions in "
			"shapes %s, %s", ashape.to_string().c_str(),
			bshape.to_string().c_str());
	}
	auto ap = permute(extend(a, 2, {bshape.at(0)}), {2, 1, 0});
	auto bp = permute(extend(b, 2, {ashape.at(1)}), {0, 2, 1});
	return reduce_sum(mul(ap, bp), 2);
}

DataNode convolute (DataNode canvas, DataNode window)
{
	throw std::bad_function_call(); // unimplemented
}

}

#endif
