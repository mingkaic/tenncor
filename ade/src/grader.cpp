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
	return Functor<ABS>::get({args.front()->gradient(wrt)});
}

NOARG_SIG(NEG)
{
	return Functor<NEG>::get({args.front()->gradient(wrt)});
}

NOARG_SIG(NOT)
{
	return Functor<NOT>::get({args.front()->gradient(wrt)});
}

NOARG_SIG(SIN)
{
	// sin'(f) = f'*cos(f)
	return Functor<MUL>::get({
		args.front()->gradient(wrt),
		Functor<COS>::get({args.front()})});
}

NOARG_SIG(COS)
{
	// cos'(f) = -f'*sin(f)
	return Functor<MUL>::get({
		Functor<NEG>::get({args.front()->gradient(wrt)}),
		Functor<SIN>::get({args.front()})});
}

NOARG_SIG(TAN)
{
	// tan'(f) = f'*sec^2(f)
	// 		= f'/cos^2(f)
	Tensorptr denom = Functor<COS>::get({args.front()});
	return Functor<DIV>::get({
		Functor<DIV>::get({
			args.front()->gradient(wrt), denom}), denom});
}

NOARG_SIG(EXP)
{
	// exp'(f) = f'*exp(f)
	return Functor<MUL>::get({
		args.front()->gradient(wrt),
		Functor<EXP>::get({args.front()})});
}

NOARG_SIG(LOG)
{
	// log'(f) = f' / f
	return Functor<DIV>::get({
		args.front()->gradient(wrt),
		 args.front()});
}

NOARG_SIG(SQRT)
{
	// sqrt'(f) = f'/(2*sqrt(f))
	Tensorptr denom = Functor<SQRT>::get({args.front()});
	return Functor<DIV>::get({
		args.front()->gradient(wrt),
		Functor<ADD>::get({denom, denom})});
}

NOARG_SIG(ROUND)
{
	// round'(f) = round(f')
	return Functor<ROUND>::get({args.front()->gradient(wrt)});
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
	// h'(f, g) = f' + g'
	return Functor<ADD>::get({
		args.front()->gradient(wrt),
		args.back()->gradient(wrt)});
}

NOARG_SIG(SUB)
{
	// h'(f, g) = f' - g'
	return Functor<SUB>::get({
		args.front()->gradient(wrt),
		args.back()->gradient(wrt)});
}

NOARG_SIG(MUL)
{
	// h'(f, g) = f' * g + g' * f
	return Functor<ADD>::get({
		Functor<MUL>::get({
			args.front(), args.front()->gradient(wrt)}),
		Functor<MUL>::get({
			args.back(), args.back()->gradient(wrt)})});
}

NOARG_SIG(DIV)
{
	// h'(f, g) = (f' * g - g' * f) / g^2
	//			= f' / g - ((g' * f) / g) / g
	Tensorptr& f = args[0];
	Tensorptr& g = args[1];
	Tensorptr df = f->gradient(wrt);
	Tensorptr dg = g->gradient(wrt);
	return Functor<SUB>::get({
		Functor<DIV>::get({df, g}),
		Functor<DIV>::get({
			Functor<DIV>::get({
				Functor<MUL>::get({dg, f}), g}), g})});
}

NOARG_SIG(EQ)
{
	return Functor<EQ>::get({
		args.front()->gradient(wrt),
		args.back()->gradient(wrt)});
}

NOARG_SIG(NE)
{
	return Functor<NE>::get({
		args.front()->gradient(wrt),
		args.back()->gradient(wrt)});
}

NOARG_SIG(LT)
{
	return Functor<LT>::get({
		args.front()->gradient(wrt),
		args.back()->gradient(wrt)});
}

NOARG_SIG(GT)
{
	return Functor<GT>::get({
		args.front()->gradient(wrt),
		args.back()->gradient(wrt)});
}

ZERO_GRAD(BINO)

ZERO_GRAD(UNIF)

ZERO_GRAD(NORM)

ZERO_GRAD(N_ELEMS)

ZERO_GRAD(N_DIMS)

NOARG_SIG(ARGMAX)
{
	// ARGMAX has no gradient
	throw std::bad_function_call();
}

NOARG_SIG(RMAX)
{
	Tensorptr& a = args[0];
	Tensorptr da = a->gradient(wrt);
	Tensorptr ismax = Functor<EQ>::get({a,
		Functor<RMAX>::get({a})});
	Tensorptr nmax = Functor<RSUM>::get({ismax});
	Tensorptr g = Functor<DIV>::get({ismax, nmax});
	return Functor<MUL>::get({g, da});
}

NOARG_SIG(RSUM)
{
	return args.front()->gradient(wrt);
}

template <>
Tensorptr grader<MATMUL> (std::vector<Tensorptr> args, Tensorptr& wrt)
{
	return grader<MATMUL,uint8_t,uint8_t>(args, wrt, 1, 1);
}

template <>
Tensorptr grader<MATMUL,uint8_t,uint8_t> (std::vector<Tensorptr> args,
	Tensorptr& wrt, uint8_t agroup_idx, uint8_t bgroup_idx)
{
	// dc(a, b)/dx = matmul(da/dx[shape:fx], dc/da[shape:ca])[shape:cx] +
	//	matmul(db/dx[shape:bx], dc/db[shape:cb])[shape:cx]
	Tensorptr& a = args[0];
	Tensorptr& b = args[1];
	Shape ashape = a->shape();
	Shape bshape = b->shape();
	uint8_t arank = ashape.n_rank();
	uint8_t brank = bshape.n_rank();
	uint8_t agroup_idx1 = arank - agroup_idx;
	uint8_t bgroup_idx1 = brank - bgroup_idx;
	auto fit = ashape.begin();
	auto git = bshape.begin();
	std::vector<DimT> a_ext(fit + agroup_idx, fit + arank); // agroup1
	std::vector<DimT> b_ext(git, git + bgroup_idx); // bgroup0

	// [bgroup0<0:bgroup_idx>, bgroup1<bgroup_idx:brank>, a_ext<brank:>]
	Tensorptr lhs = b;
	if (a_ext.size() > 0)
	{
		lhs = Functor<EXTEND,std::vector<DimT>>::get({lhs}, a_ext);
	}
	uint8_t lrank = lhs->shape().n_rank();
	uint8_t n_aext = a_ext.size();
	// [bgroup0, a_ext, bgroup1, a_ext]
	std::vector<uint8_t> lindices(lrank + n_aext);
	for (uint8_t i = 0; i < bgroup_idx; ++i)
	{
		lindices[i] = i;
	}
	for (uint8_t i = 0; i < bgroup_idx1; ++i)
	{
		lindices[bgroup_idx + n_aext + i] = bgroup_idx + i;
	}
	for (uint8_t i = 0; i < n_aext; ++i)
	{
		lindices[bgroup_idx + i] = brank + i;
		lindices[lrank + i] = brank + i;
	}
	lhs = Functor<PERMUTE,std::vector<uint8_t>>::get({lhs}, lindices);

	// [agroup0<0:agroup_idx>, agroup1<agroup_idx:arank>, b_ext<arank:>]
	Tensorptr rhs = a;
	if (b_ext.size() > 0)
	{
		rhs = Functor<EXTEND,std::vector<DimT>>::get({rhs}, b_ext);
	}
	uint8_t rrank = rhs->shape().n_rank();
	uint8_t n_bext = b_ext.size();
	// [b_ext, agroup1, b_ext, agroup0]
	std::vector<uint8_t> rindices(rrank + n_bext);
	for (uint8_t i = 0; i < agroup_idx; ++i)
	{
		rindices[2 * n_bext + agroup_idx1 + i] = i;
	}
	for (uint8_t i = 0; i < agroup_idx1; ++i)
	{
		rindices[agroup_idx + i] = agroup_idx + i;
	}
	for (uint8_t i = 0; i < n_bext; ++i)
	{
		rindices[i] = arank + i;
		rindices[n_bext + agroup_idx1 + i] = arank + i;
	}
	rhs = Functor<PERMUTE,std::vector<uint8_t>>::get({rhs}, rindices);

	const Shape& wrtshape = wrt->shape();
	uint8_t wrank = wrtshape.n_rank();
	auto wit = wrtshape.begin();

	uint8_t lgroup = bgroup_idx + n_aext;
	Tensorptr dlhs = a->gradient(wrt);
	const Shape& dlshape = dlhs->shape();
	auto dlit = dlshape.begin();
	uint8_t dlrank = dlshape.n_rank();
	if (std::equal(dlit + lgroup, dlit + dlrank, wit) &&
		dlrank - lgroup == wrank)
	{
		lhs = Functor<MATMUL,uint8_t,uint8_t>::get({dlhs, lhs}, lgroup, lgroup);
	}

	uint8_t rgroup = n_bext + agroup_idx1;
	Tensorptr drhs = b->gradient(wrt);
	const Shape& drshape = drhs->shape();
	auto drit = drshape.begin();
	uint8_t drrank = drshape.n_rank();
	if (std::equal(drit + rgroup, drit + drshape.n_rank(), wit) &&
		drrank - rgroup == wrank)
	{
		rhs = Functor<MATMUL,uint8_t,uint8_t>::get({drhs, rhs}, rgroup, rgroup);
	}

	if (lhs->shape().compatible_after(rhs->shape(), 0))
	{
		return Functor<ADD>::get({lhs, rhs});
	}
	if (std::equal(wit, wit + wrank,
		lhs->shape().begin() + bgroup_idx1 + n_aext))
	{
		return lhs;
	}
	return rhs;
}

template <>
Tensorptr grader<PERMUTE,std::vector<uint8_t>> (
	std::vector<Tensorptr> args, Tensorptr& wrt, std::vector<uint8_t>)
{
	return args.front()->gradient(wrt);
}

SHPARG_SIG(EXTEND)
{
	return Functor<EXTEND,std::vector<DimT>>::get(
		{args.front()->gradient(wrt)}, shape);
}

SHPARG_SIG(RESHAPE)
{
	return Functor<RESHAPE,std::vector<DimT>>::get(
		{args.front()->gradient(wrt)}, shape);
}

#undef NOARG_SIG

#undef INTARG_SIG

#undef SHPARG_SIG

#undef ZERO_GRAD

}

#endif
