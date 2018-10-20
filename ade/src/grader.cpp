#include "ade/grader.hpp"
#include "ade/tensor.hpp"
#include "ade/functor.hpp"

namespace ade
{

#ifdef ADE_GRADER_HPP

#define NOARG_SIG(CODE)template <>\
Tensorptr grader<CODE> (std::vector<Tensorptr> args, Tensorptr& wrt)

#define INTARG_SIG(CODE)template <>\
Tensorptr grader<CODE> (std::vector<Tensorptr> args, Tensorptr& wrt,\
	uint8_t dim)

#define SHPARG_SIG(CODE)template <>\
Tensorptr grader<CODE,std::vector<DimT>> (std::vector<Tensorptr> args,\
	Tensorptr& wrt, std::vector<DimT> shape)

#define ZERO_GRAD(CODE)NOARG_SIG(CODE)\
{ return constant_zero(wrt->shape().as_list()); }

NOARG_SIG(ABS)
{
	// abs'(f) = f * f' / abs(f)
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<DIV>::get({Functor<MUL>::get({args.front(), df}),
		Functor<ABS>::get({args.front()})});
}

NOARG_SIG(NEG)
{
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<NEG>::get({df});
}

NOARG_SIG(NOT)
{
	return Functor<NOT>::get({args.front()->gradient(wrt)});
}

NOARG_SIG(SIN)
{
	// sin'(f) = f'*cos(f)
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<MUL>::get({df, Functor<COS>::get({args.front()})});
}

NOARG_SIG(COS)
{
	// cos'(f) = -f'*sin(f)
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<MUL>::get({Functor<NEG>::get({df}),
		Functor<SIN>::get({args.front()})});
}

NOARG_SIG(TAN)
{
	// tan'(f) = f'*sec^2(f)
	// 		= f'/cos^2(f)
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	Tensorptr denom = Functor<COS>::get({args.front()});
	return Functor<DIV>::get({Functor<DIV>::get({df, denom}), denom});
}

NOARG_SIG(EXP)
{
	// exp'(f) = f'*exp(f)
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<MUL>::get({df, Functor<EXP>::get({args.front()})});
}

NOARG_SIG(LOG)
{
	// log'(f) = f' / f
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<DIV>::get({df, args.front()});
}

NOARG_SIG(SQRT)
{
	// sqrt'(f) = f'/(2*sqrt(f))
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	Tensorptr denom = Functor<SQRT>::get({args.front()});
	return Functor<DIV>::get({df, Functor<ADD>::get({denom, denom})});
}

NOARG_SIG(ROUND)
{
	// round'(f) = round(f')
	auto df = args.front()->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<ROUND>::get({df});
}

NOARG_SIG(FLIP)
{
	return args.front()->gradient(wrt);
}

NOARG_SIG(POW)
{
	// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
	//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
	Tensorptr& f = args[0];
	Tensorptr& g = args[1];
	Tensorptr df = f->gradient(wrt);
	Tensorptr dg = g->gradient(wrt);
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get() &&
		dg.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	return Functor<MUL>::get({
		Functor<POW>::get({
			f, Functor<SUB>::get({g, Tensor::SYMBOLIC_ONE})}),
		Functor<ADD>::get({
			Functor<MUL>::get({df, g}),
			Functor<MUL>::get({dg,
				Functor<MUL>::get({f, Functor<LOG>::get({f})})
			})
		})
	});
}

NOARG_SIG(ADD)
{
	// h'(f, g, ...) = f' + g' + ...
	std::vector<Tensorptr> gargs;
	std::transform(args.begin(), args.end(), std::back_inserter(gargs),
	[&](ade::Tensorptr& arg)
	{
		return arg->gradient(wrt);
	});
	std::vector<Tensorptr> filtered;
	std::copy_if(gargs.begin(), gargs.end(), std::back_inserter(filtered),
		[](Tensorptr& d)
		{
			return d.get() != ade::Tensor::SYMBOLIC_ZERO.get();
		});
	if (filtered.empty())
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	if (1 == filtered.size())
	{
		return filtered[0];
	}
	return Functor<ADD>::get(filtered);
}

NOARG_SIG(SUB)
{
	// h'(f, g) = f' - g'
	Tensorptr df = args[0]->gradient(wrt);
	Tensorptr dg = args[1]->gradient(wrt);
	if (dg.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return Functor<NEG>::get({dg});
	}
	return Functor<SUB>::get({df, dg});
}

NOARG_SIG(MUL)
{
	// h'(f, g) = f' * g * ... + f * g' * ... + ...
	std::vector<Tensorptr> gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		// f' * g * ...
		Tensorptr f = args[i];
		args[i] = args[i]->gradient(wrt);
		gargs.push_back(Functor<MUL>::get(args));
		args[i] = f;
	}
	if (std::any_of(gargs.begin(), gargs.end(),
		[](Tensorptr& d)
		{
			return d.get() == ade::Tensor::SYMBOLIC_ZERO.get();
		}))
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	return Functor<ADD>::get(gargs);
}

NOARG_SIG(DIV)
{
	// h'(f, g) = (f' * g - g' * f) / g^2
	//			= f' / g - ((g' * f) / g) / g
	Tensorptr& f = args[0];
	Tensorptr& g = args[1];
	Tensorptr df = f->gradient(wrt);
	Tensorptr dg = g->gradient(wrt);
	if (df.get() != ade::Tensor::SYMBOLIC_ZERO.get())
	{
		df = Functor<DIV>::get({df, g});
	}
	if (dg.get() != ade::Tensor::SYMBOLIC_ZERO.get())
	{
		dg = Functor<DIV>::get({Functor<DIV>::get({
			Functor<MUL>::get({dg, f}), g}), g});
	}
	if (dg.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return df;
	}
	if (df.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return Functor<NEG>::get({dg});
	}
	return Functor<SUB>::get({df, dg});
}

NOARG_SIG(EQ)
{
	Tensorptr df = args[0]->gradient(wrt);
	Tensorptr dg = args[1]->gradient(wrt);
	if (df.get() == dg.get())
	{
		return Tensor::SYMBOLIC_ONE;
	}
	return Functor<EQ>::get({df, dg});
}

NOARG_SIG(NE)
{
	Tensorptr df = args[0]->gradient(wrt);
	Tensorptr dg = args[1]->gradient(wrt);
	if (df.get() == dg.get())
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	return Functor<NE>::get({df, dg});
}

NOARG_SIG(LT)
{
	Tensorptr df = args[0]->gradient(wrt);
	Tensorptr dg = args[1]->gradient(wrt);
	if (df.get() == dg.get())
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	return Functor<LT>::get({df, dg});
}

NOARG_SIG(GT)
{
	Tensorptr df = args[0]->gradient(wrt);
	Tensorptr dg = args[1]->gradient(wrt);
	if (df.get() == dg.get())
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	return Functor<GT>::get({df, dg});
}

NOARG_SIG(MIN)
{
	Tensorptr f = Functor<MIN>::get(args);
	std::vector<Tensorptr> gargs;
	std::transform(args.begin(), args.end(), std::back_inserter(gargs),
	[&](ade::Tensorptr& arg)
	{
		return Functor<MUL>::get({
			Functor<EQ>::get({f, arg}),
			arg->gradient(wrt)});
	});
	if (std::all_of(gargs.begin(), gargs.end(),
		[](Tensorptr& d)
		{
			return d.get() == ade::Tensor::SYMBOLIC_ZERO.get();
		}))
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	return Functor<ADD>::get(gargs);
}

NOARG_SIG(MAX)
{
	Tensorptr f = Functor<MAX>::get(args);
	std::vector<Tensorptr> gargs;
	std::transform(args.begin(), args.end(), std::back_inserter(gargs),
	[&](ade::Tensorptr& arg)
	{
		return Functor<MUL>::get({
			Functor<EQ>::get({f, arg}),
			arg->gradient(wrt)});
	});
	if (std::all_of(gargs.begin(), gargs.end(),
		[](Tensorptr& d)
		{
			return d.get() == ade::Tensor::SYMBOLIC_ZERO.get();
		}))
	{
		return Tensor::SYMBOLIC_ZERO;
	}
	return Functor<ADD>::get(gargs);
}

ZERO_GRAD(RAND_BINO)

ZERO_GRAD(RAND_UNIF)

ZERO_GRAD(RAND_NORM)

ZERO_GRAD(N_ELEMS)

ZERO_GRAD(N_DIMS)

INTARG_SIG(ARGMAX)
{
	// ARGMAX has no gradient
	throw std::bad_function_call();
}

INTARG_SIG(RMAX)
{
	Tensorptr& a = args[0];
	Tensorptr da = a->gradient(wrt);
	if (da.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return da;
	}
	Tensorptr ismax = Functor<EQ>::get({a,
		Functor<RMAX,uint8_t>::get({a}, dim)});
	Tensorptr nmax = Functor<RSUM,uint8_t>::get({ismax}, dim);
	Tensorptr g = Functor<DIV>::get({ismax, nmax});
	return Functor<MUL>::get({g, da});
}

INTARG_SIG(RSUM)
{
	return args.front()->gradient(wrt);
}

template <>
Tensorptr grader<MATMUL,uint8_t,uint8_t> (std::vector<Tensorptr> args,
	Tensorptr& wrt, uint8_t agroup_idx, uint8_t bgroup_idx)
{
	// dc(a, b)/dx =
	//	matmul(da/dx[shape:fx], dc/da[shape:ca])[shape:cx] +
	//	matmul(db/dx[shape:bx], dc/db[shape:cb])[shape:cx]
	Tensorptr& a = args[0];
	Tensorptr& b = args[1];
	Shape ashape = a->shape();
	Shape bshape = b->shape();
	uint8_t arank = ashape.n_rank();
	uint8_t brank = bshape.n_rank();
	uint8_t agroup_idx1 = arank - agroup_idx;
	uint8_t bgroup_idx1 = brank - bgroup_idx;
	uint8_t crank = bgroup_idx + agroup_idx1;
	auto ita = ashape.begin();
	auto itb = bshape.begin();
	std::vector<DimT> agroup1(ita + agroup_idx, ita + arank); // agroup1
	std::vector<DimT> bgroup0(itb, itb + bgroup_idx); // bgroup0

	// [bgroup0<0:bgroup_idx>, bgroup1<bgroup_idx:brank>, agroup1<brank:>]
	Tensorptr lhs = b;
	if (agroup1.size() > 0)
	{
		lhs = Functor<EXTEND,std::vector<DimT>>::get({lhs}, agroup1);
	}
	// [bgroup0, agroup1, bgroup1, agroup1]
	std::vector<uint8_t> lindices(crank + arank);
	{
		auto it = lindices.begin();
		std::iota(it, it + bgroup_idx, 0); // bgroup0
		std::iota(it + bgroup_idx, it + crank, brank); // agroup1
		std::iota(it + crank, it + crank + bgroup_idx1, bgroup_idx); // bgroup1
		std::iota(it + crank + bgroup_idx1, lindices.end(), brank); // agroup1
	}
	lhs = Functor<PERMUTE,std::vector<uint8_t>>::get({lhs}, lindices);

	// [agroup0<0:agroup_idx>, agroup1<agroup_idx:arank>, bgroup0<arank:>]
	Tensorptr rhs = a;
	if (bgroup0.size() > 0)
	{
		rhs = Functor<EXTEND,std::vector<DimT>>::get({rhs}, bgroup0);
	}
	// [bgroup0, agroup1, bgroup0, agroup0]
	std::vector<uint8_t> rindices(crank + brank);
	{
		auto it = rindices.begin();
		std::iota(it, it + bgroup_idx, arank); // bgroup0
		std::iota(it + bgroup_idx, it + crank, agroup_idx); // agroup1
		std::iota(it + crank, it + crank + bgroup_idx, arank); // bgroup0
		std::iota(it + crank + bgroup_idx, rindices.end(), 0); // agroup0
	}
	rhs = Functor<PERMUTE,std::vector<uint8_t>>::get({rhs}, rindices);

	const Shape& wrtshape = wrt->shape();
	uint8_t wrank = wrtshape.n_rank();
	auto wit = wrtshape.begin();

	Tensorptr dlhs = a->gradient(wrt);
	const Shape& dlshape = dlhs->shape();
	auto dlit = dlshape.begin();
	uint8_t dlrank = dlshape.n_rank();
	if (dlhs.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		lhs = ade::Tensor::SYMBOLIC_ZERO;
	}
	else if (std::equal(dlit + crank, dlit + dlrank, wit) &&
		dlrank - crank == wrank)
	{
		lhs = Functor<MATMUL,uint8_t,uint8_t>::get({dlhs, lhs}, crank, crank);
	}

	Tensorptr drhs = b->gradient(wrt);
	const Shape& drshape = drhs->shape();
	auto drit = drshape.begin();
	uint8_t drrank = drshape.n_rank();
	if (drhs.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		rhs = ade::Tensor::SYMBOLIC_ZERO;
	}
	else if (std::equal(drit + crank, drit + drshape.n_rank(), wit) &&
		drrank - crank == wrank)
	{
		rhs = Functor<MATMUL,uint8_t,uint8_t>::get({drhs, rhs}, crank, crank);
	}

	if (lhs.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return rhs;
	}
	if (rhs.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return lhs;
	}
	return Functor<ADD>::get({lhs, rhs});
}

template <>
Tensorptr grader<PERMUTE,std::vector<uint8_t>> (
	std::vector<Tensorptr> args, Tensorptr& wrt, std::vector<uint8_t>)
{
	return args.front()->gradient(wrt);
}

SHPARG_SIG(EXTEND)
{
	Tensorptr da = args.front()->gradient(wrt);
	if (da.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return da;
	}
	return Functor<EXTEND,std::vector<DimT>>::get({da}, shape);
}

SHPARG_SIG(RESHAPE)
{
	Tensorptr da = args.front()->gradient(wrt);
	if (da.get() == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		return da;
	}
	return Functor<RESHAPE,std::vector<DimT>>::get({da}, shape);
}

#undef NOARG_SIG

#undef INTARG_SIG

#undef SHPARG_SIG

#undef ZERO_GRAD

}

#endif
