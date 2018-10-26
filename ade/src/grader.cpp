#include "ade/functor.hpp"

namespace ade
{

#ifdef ADE_GRADER_HPP

#define GRAD_SIGNATURE(CODE)template <>\
Tensorptr grader<CODE> (Tensorptr& fwd, ArgsT& args,\
	std::vector<Tensorptr>& grads)

#define ZERO_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ return shaped_zero(fwd->shape()); }

static void check_unary (const char* op, ArgsT& args)
{
	if (1 != args.size())
	{
		fatalf("cannot %s for non-single argument(s): using %d argument(s)",
			op, args.size());
	}
}

static void check_binary (const char* op, ArgsT& args)
{
	if (2 != args.size())
	{
		fatalf("cannot %s for non-binary argument(s): using %d argument(s)",
			op, args.size());
	}
}

static void check_nnary (const char* op, ArgsT& args)
{
	if (0 == args.size())
	{
		fatalf("cannot %s with arguments", op);
	}
}

GRAD_SIGNATURE(COPY)
{
	check_unary("COPY", args);
	auto child = args.front();
	return Functor<COPY>::get({{child.first, grads.front()}});
}

GRAD_SIGNATURE(ABS)
{
	check_unary("ABS", args);
	auto child = args.front();
	// abs'(f) = f * f' / abs(f)
	return Functor<DIV>::get({
		{identity, Functor<MUL>::get({child, {child.first, grads.front()}})},
		{identity, fwd},
	});
}

GRAD_SIGNATURE(NEG)
{
	check_unary("NEG", args);
	// neg'(f) = -f'
	return Functor<NEG>::get({{args.front().first, grads.front()}});
}

GRAD_SIGNATURE(NOT)
{
	check_unary("NOT", args);
	// neg'(f) = not(f')
	return Functor<NOT>::get({{args.front().first, grads.front()}});
}

GRAD_SIGNATURE(SIN)
{
	check_unary("SIN", args);
	auto child = args.front();
	// sin'(f) = f'*cos(f)
	return Functor<MUL>::get({
		{child.first, grads.front()},
		{identity, Functor<COS>::get({child})},
	});
}

GRAD_SIGNATURE(COS)
{
	check_unary("COS", args);
	auto child = args.front();
	// cos'(f) = -f'*sin(f)
	return Functor<MUL>::get({
		{identity, Functor<NEG>::get({{child.first, grads.front()}})},
		{identity, Functor<SIN>::get({child})},
	});
}

GRAD_SIGNATURE(TAN)
{
	check_unary("TAN", args);
	auto child = args.front();
	// tan'(f) = f'*sec^2(f)
	// 		= f'/cos^2(f)
	Tensorptr denom = Functor<COS>::get({child});
	return Functor<DIV>::get({
		{identity, Functor<DIV>::get({
			{child.first, grads.front()},
			{identity, denom}
		})},
		{identity, denom},
	});
}

GRAD_SIGNATURE(EXP)
{
	check_unary("EXP", args);
	auto child = args.front();
	// exp'(f) = f'*exp(f)
	return Functor<MUL>::get({
		{child.first, grads.front()},
		{identity, fwd},
	});
}

GRAD_SIGNATURE(LOG)
{
	check_unary("LOG", args);
	auto child = args.front();
	// log'(f) = f' / f
	return Functor<DIV>::get({{child.first, grads.front()}, child});
}

GRAD_SIGNATURE(SQRT)
{
	check_unary("SQRT", args);
	auto child = args.front();
	// sqrt'(f) = f'/(2*sqrt(f))
	return Functor<DIV>::get({
		{child.first, grads.front()},
		{identity, Functor<ADD>::get({
			{identity, fwd},
			{identity, fwd}
		})},
	});
}

GRAD_SIGNATURE(ROUND)
{
	check_unary("ROUND", args);
	auto child = args.front();
	// round'(f) = round(f')
	return Functor<ROUND>::get({{child.first, grads.front()}});
}

GRAD_SIGNATURE(POW)
{
	check_binary("POW", args);
	// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
	//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
	auto& child_f = args[0];
	auto& child_g = args[1];
	Tensorptr lhs = Functor<ADD>::get({
		{identity, Functor<MUL>::get({
			{child_f.first, grads[0]},
			child_g
		})},
		{identity, Functor<MUL>::get({
			{child_g.first, grads[1]},
			child_f,
			{identity, Functor<LOG>::get({child_f})},
		})},
	});
	return Functor<MUL>::get({
		{identity, Functor<POW>::get({
			child_f,
			{identity, Functor<SUB>::get({
				child_g,
				{identity, shaped_one(child_g.second->shape())},
			})},
		})},
		{identity, lhs}
	});
}

GRAD_SIGNATURE(ADD)
{
	check_nnary("ADD", args);
	// h'(f, g, ...) = f' + g' + ...
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		gargs.push_back({args[i].first, grads[i]});
	};
	return Functor<ADD>::get(gargs);
}

GRAD_SIGNATURE(SUB)
{
	check_binary("SUB", args);
	// h'(f, g) = f' - g'
	return Functor<SUB>::get({
		{args[0].first, grads[0]},
		{args[1].first, grads[1]}
	});
}

GRAD_SIGNATURE(MUL)
{
	check_nnary("MUL", args);
	// h'(f, g) = f' * g * ... + f * g' * ... + ...
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		// f' * g * ...
		auto child = args[i];
		args[i] = {child.first, grads[i]};
		gargs.push_back({identity, Functor<MUL>::get(args)});
		args[i] = child;
	}
	return Functor<ADD>::get(gargs);
}

GRAD_SIGNATURE(DIV)
{
	check_binary("DIV", args);
	// h'(f, g) = (f' * g - g' * f) / g^2
	//			= f' / g - ((g' * f) / g) / g
	auto& child_f = args[0];
	auto& child_g = args[1];
	return Functor<SUB>::get({
		{identity, Functor<DIV>::get({
			{child_f.first, grads[0]},
			child_g,
		})},
		{identity, Functor<DIV>::get({
			{identity, Functor<DIV>::get({
				{identity, Functor<MUL>::get({
					{child_g.first, grads[1]},
					child_f,
				})},
				child_g,
			})},
			child_g,
		})},
	});
}

GRAD_SIGNATURE(MIN)
{
	check_nnary("MIN", args);
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		gargs.push_back({identity, Functor<MUL>::get({
			{identity, Functor<EQ>::get({
				{identity, fwd},
				args[i]
			})},
			{args[i].first, grads[i]},
		})});
	}
	return Functor<ADD>::get(gargs);
}

GRAD_SIGNATURE(MAX)
{
	check_nnary("MAX", args);
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		gargs.push_back({identity, Functor<MUL>::get({
			{identity, Functor<EQ>::get({
				{identity, fwd},
				args[i]
			})},
			{args[i].first, grads[i]},
		})});
	}
	return Functor<ADD>::get(gargs);
}

ZERO_GRAD(EQ)

ZERO_GRAD(NE)

ZERO_GRAD(LT)

ZERO_GRAD(GT)

ZERO_GRAD(RAND_BINO)

ZERO_GRAD(RAND_UNIF)

ZERO_GRAD(RAND_NORM)

#undef GRAD_SIGNATURE

#undef ZERO_GRAD

#define CALL_GRAD(OP)case OP: return grader<OP>(fwd, args, grads);

Tensorptr gradmap (OPCODE op, Tensorptr& fwd, ArgsT args,
	std::vector<Tensorptr>& grads)
{
	switch (op)
	{
		CALL_GRAD(COPY)
		CALL_GRAD(ABS)
		CALL_GRAD(NEG)
		CALL_GRAD(NOT)
		CALL_GRAD(SIN)
		CALL_GRAD(COS)
		CALL_GRAD(TAN)
		CALL_GRAD(EXP)
		CALL_GRAD(LOG)
		CALL_GRAD(SQRT)
		CALL_GRAD(ROUND)
		CALL_GRAD(POW)
		CALL_GRAD(ADD)
		CALL_GRAD(SUB)
		CALL_GRAD(MUL)
		CALL_GRAD(DIV)
		CALL_GRAD(EQ)
		CALL_GRAD(NE)
		CALL_GRAD(LT)
		CALL_GRAD(GT)
		CALL_GRAD(MIN)
		CALL_GRAD(MAX)
		CALL_GRAD(RAND_BINO)
		CALL_GRAD(RAND_UNIF)
		CALL_GRAD(RAND_NORM)
		default: break;
	}
	return Tensorptr(nullptr);
}

}

#endif
