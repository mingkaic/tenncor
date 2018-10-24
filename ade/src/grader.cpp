#include "ade/grader.hpp"

namespace ade
{

#ifdef ADE_GRADER_HPP

#define GRAD_SIGNATURE(CODE)template <>\
Tensorptr grader<CODE> (Tensorptr& fwd, ArgsT& args, Tensorptr& wrt)

#define ZERO_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ return shaped_zero(wrt->shape()); }

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
	return Functor<COPY>::get({{child.first, child.second->gradient(wrt)}});
}

GRAD_SIGNATURE(ABS)
{
	check_unary("ABS", args);
	auto child = args.front();
	// abs'(f) = f * f' / abs(f)
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<DIV>::get({
		{identity, Functor<MUL>::get({child, {mapper, df}})},
		{identity, fwd},
	});
}

GRAD_SIGNATURE(NEG)
{
	check_unary("NEG", args);
	auto child = args.front();
	// neg'(f) = -f'
	return Functor<NEG>::get({{child.first, child.second->gradient(wrt)}});
}

GRAD_SIGNATURE(NOT)
{
	check_unary("NOT", args);
	auto child = args.front();
	// neg'(f) = not(f')
	return Functor<NOT>::get({{child.first, child.second->gradient(wrt)}});
}

GRAD_SIGNATURE(SIN)
{
	check_unary("SIN", args);
	auto child = args.front();
	// sin'(f) = f'*cos(f)
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<MUL>::get({
		{mapper, df},
		{identity, Functor<COS>::get({child})},
	});
}

GRAD_SIGNATURE(COS)
{
	check_unary("COS", args);
	auto child = args.front();
	// cos'(f) = -f'*sin(f)
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<MUL>::get({
		{identity, Functor<NEG>::get({{mapper, df}})},
		{identity, Functor<SIN>::get({child})},
	});
}

GRAD_SIGNATURE(TAN)
{
	check_unary("TAN", args);
	auto child = args.front();
	// tan'(f) = f'*sec^2(f)
	// 		= f'/cos^2(f)
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	Tensorptr denom = Functor<COS>::get({child});
	return Functor<DIV>::get({
		{identity, Functor<DIV>::get({
			{mapper, df},
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
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<MUL>::get({
		{mapper, df},
		{identity, fwd},
	});
}

GRAD_SIGNATURE(LOG)
{
	check_unary("LOG", args);
	auto child = args.front();
	// log'(f) = f' / f
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<DIV>::get({{mapper, df}, child});
}

GRAD_SIGNATURE(SQRT)
{
	check_unary("SQRT", args);
	auto child = args.front();
	// sqrt'(f) = f'/(2*sqrt(f))
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<DIV>::get({
		{mapper, df},
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
	auto mapper = child.first;
	auto df = child.second->gradient(wrt);
	return Functor<ROUND>::get({{mapper, df}});
}

GRAD_SIGNATURE(FLIP) // todo: fix this
{
	check_unary("FLIP", args);
	return args.front().second->gradient(wrt);
}

GRAD_SIGNATURE(POW)
{
	check_binary("POW", args);
	// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
	//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
	auto& child_f = args[0];
	auto& child_g = args[1];
	auto mapper_f = child_f.first;
	auto mapper_g = child_g.first;
	Tensorptr df = child_f.second->gradient(wrt);
	Tensorptr dg = child_g.second->gradient(wrt);
	Tensorptr lhs = Functor<ADD>::get({
		{identity, Functor<MUL>::get({
			{mapper_f, df},
			child_g
		})},
		{identity, Functor<MUL>::get({
			{mapper_g, dg},
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
	std::transform(args.begin(), args.end(), std::back_inserter(gargs),
	[&](std::pair<CoordPtrT,Tensorptr>& arg) -> std::pair<CoordPtrT,Tensorptr>
	{
		return {arg.first, arg.second->gradient(wrt)};
	});
	return Functor<ADD>::get(gargs);
}

GRAD_SIGNATURE(SUB)
{
	check_binary("SUB", args);
	// h'(f, g) = f' - g'
	return Functor<SUB>::get({
		{args[0].first, args[0].second->gradient(wrt)},
		{args[1].first, args[1].second->gradient(wrt)}
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
		args[i] = {child.first, child.second->gradient(wrt)};
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
	auto mapper_f = child_f.first;
	auto mapper_g = child_g.first;
	Tensorptr df = child_f.second->gradient(wrt);
	Tensorptr dg = child_g.second->gradient(wrt);
	return Functor<SUB>::get({
		{identity, Functor<DIV>::get({
			{mapper_f, df},
			child_g,
		})},
		{identity, Functor<DIV>::get({
			{identity, Functor<DIV>::get({
				{identity, Functor<MUL>::get({
					{mapper_g, dg},
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
	std::transform(args.begin(), args.end(), std::back_inserter(gargs),
	[&](std::pair<CoordPtrT,Tensorptr>& arg) -> std::pair<CoordPtrT,Tensorptr>
	{
		return {identity, Functor<MUL>::get({
			{identity, Functor<EQ>::get({
				{identity, fwd},
				arg
			})},
			{arg.first, arg.second->gradient(wrt)},
		})};
	});
	return Functor<ADD>::get(gargs);
}

GRAD_SIGNATURE(MAX)
{
	check_nnary("MAX", args);
	ArgsT gargs;
	std::transform(args.begin(), args.end(), std::back_inserter(gargs),
	[&](std::pair<CoordPtrT,Tensorptr>& arg) -> std::pair<CoordPtrT,Tensorptr>
	{
		return {identity, Functor<MUL>::get({
			{identity, Functor<EQ>::get({
				{identity, fwd},
				arg
			})},
			{arg.first, arg.second->gradient(wrt)},
		})};
	});
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

}

#endif
