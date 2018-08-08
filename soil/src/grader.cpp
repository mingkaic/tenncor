#include "soil/grader.hpp"
#include "soil/external.hpp"
#include "soil/constant.hpp"

#include "util/mapper.hpp"

#ifdef SOIL_GRADER_HPP

Nodeptr fwd_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return args.front()->gradient(wrt);
}

Nodeptr zero_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return get_zero(Meta{wrt->shape(), wrt->type()});
}

Nodeptr abs_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return abs(args.front()->gradient(wrt));
}

Nodeptr neg_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return -args.front()->gradient(wrt);
}

Nodeptr logic_not_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return !args.front()->gradient(wrt);
}

Nodeptr sin_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// sin'(f) = f'*cos(f)
	return args.front()->gradient(wrt) * cos(args.front());
}

Nodeptr cos_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// cos'(f) = -f'*sin(f)
	return -args.front()->gradient(wrt) * sin(args.front());
}

Nodeptr tan_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// tan'(f) = f'*sec^2(f)
	// 		= f'/cos^2(f)
	Nodeptr denom = cos(args.front());
	return args.front()->gradient(wrt) / denom / denom;
}

Nodeptr exp_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// exp'(f) = f'*exp(f)
	return args.front()->gradient(wrt) * exp(args.front());
}

Nodeptr log_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// log'(f) = f' / f
	return args.front()->gradient(wrt) / args.front();
}

Nodeptr sqrt_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// sqrt'(f) = f'/(2*sqrt(f))
	Nodeptr denom = sqrt(args.front());
	return args.front()->gradient(wrt) / (denom + denom);
}

Nodeptr round_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// round'(f) = round(f')
	return round(args.front()->gradient(wrt));
}

Nodeptr arg_max_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	throw std::bad_function_call();
}

Nodeptr rmax_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	Nodeptr& a = args[0];
	Nodeptr da = a->gradient(wrt);
	Nodeptr ismax = a == reduce_max(a);
	Nodeptr nmax = reduce_sum(ismax);
	Nodeptr g = ismax / nmax;
	return g / da;
}


Nodeptr pow_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
	//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
	Nodeptr& f = args[0];
	Nodeptr& g = args[1];
	Nodeptr df = f->gradient(wrt);
	Nodeptr dg = g->gradient(wrt);
	return pow(f, g - get_one(Meta{g->shape(), g->type()})) *
		(df * g + dg * f * log(f));
}

Nodeptr add_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// h'(f, g) = f' + g'
	return args.front()->gradient(wrt) + args.back()->gradient(wrt);
}

Nodeptr sub_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// h'(f, g) = f' - g'
	return args.front()->gradient(wrt) - args.back()->gradient(wrt);
}

Nodeptr mul_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// h'(f, g) = f' * g + g' * f
	return args.front() * args.back()->gradient(wrt) +
		args.back() * args.front()->gradient(wrt);
}

Nodeptr div_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// h'(f, g) = (f' * g - g' * f) / g^2
	//			= f' / g - ((g' * f) / g) / g
	Nodeptr& f = args[0];
	Nodeptr& g = args[1];
	Nodeptr df = f->gradient(wrt);
	Nodeptr dg = g->gradient(wrt);
	return df / g - ((dg * f) / g) / g;
}

Nodeptr eq_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return args.front()->gradient(wrt) == args.back()->gradient(wrt);
}

Nodeptr neq_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return args.front()->gradient(wrt) != args.back()->gradient(wrt);
}

Nodeptr lt_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return args.front()->gradient(wrt) < args.back()->gradient(wrt);
}

Nodeptr gt_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	return args.front()->gradient(wrt) > args.back()->gradient(wrt);
}

Nodeptr matmul_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// dh(f, g)/dx =
	//			matmul(df/dx[shape:fx],
	//				dh/df[shape:hf])[shape:hx] +
	//			matmul(dg/dx[shape:gx],
	//				dh/dg[shape:hg])[shape:hx]
	Nodeptr& f = args[0];
	Nodeptr& g = args[1];
	Shape fshape = f->shape();
	Shape gshape = g->shape();
	NElemT fdim = fshape.group(1).n_elems();
	NElemT gdim = gshape.group(0).n_elems();
	// dh/df = transpose_0,n_rank-1(
	//		id[shape:g1g0g...f1f1] * transpose(g)[g0g1g...])[shape:gf0f1])
	Nodeptr lhs = get_identity(fdim, g->type(), gshape) * g;
	lhs = transpose(lhs, {1, lhs->shape().n_rank() - 2});
	lhs = group(lhs, {0, 2});
	// dh/dg = transpose_1,n_rank-2(
	//		id[shape:f1f0f...g1g1] * transpose(f)[f0f1f...])[shape:fg0g1])
	Nodeptr rhs = get_identity(gdim, f->type(), fshape) * f;
	rhs = transpose(rhs, {0, rhs->shape().n_rank() - 1});
	rhs = group(rhs, {0, 2});
	// todo: attempt to get rid of these checks
	if (f.get() != wrt.get())
	{
		// df/dx
		Nodeptr dlhs = f->gradient(wrt);
		if (false == higher_order(dlhs->shape()))
		{
			dlhs = group(dlhs);
		}
		lhs = matmul(dlhs, lhs);
	}
	if (g.get() != wrt.get())
	{
		// dg/dx
		Nodeptr drhs = g->gradient(wrt);
		if (false == higher_order(drhs->shape()))
		{
			drhs = group(drhs);
		}
		rhs = matmul(drhs, rhs);
	}

	return lhs + rhs;
}

static EnumMap<OPCODE,Grader> graders =
{
	{ABS, abs_grad},
	{NEG, neg_grad},
	{NOT, logic_not_grad},
	{SIN, sin_grad},
	{COS, cos_grad},
	{TAN, tan_grad},
	{EXP, exp_grad},
	{LOG, log_grad},
	{SQRT, sqrt_grad},
	{ROUND, round_grad},
	{FLIP, fwd_grad},
	{TRANSPOSE, fwd_grad},
	{N_ELEMS, zero_grad},
	{N_DIMS, zero_grad},

	{ARGMAX, arg_max_grad},
	{RMAX, rmax_grad},
	{RSUM, fwd_grad},

	{POW, pow_grad},
	{ADD, add_grad},
	{SUB, sub_grad},
	{MUL, mul_grad},
	{DIV, div_grad},
	{EQ, eq_grad},
	{NE, neq_grad},
	{LT, lt_grad},
	{GT, gt_grad},
	{BINO, zero_grad},
	{UNIF, zero_grad},
	{NORM, zero_grad},
	{MATMUL, matmul_grad},
};

Grader get_grader (OPCODE opcode)
{
	auto it = graders.find(opcode);
	if (graders.end() == it)
	{
		handle_error("failed to retrieve grader",
			ErrArg<std::string>("opcode", opname(opcode)));
	}
	return it->second;
}

#endif
