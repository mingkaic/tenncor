#include "soil/grader.hpp"
#include "soil/constant.hpp"
#include "soil/mapper.hpp"

#ifdef GRADER_HPP

Nodeptr add_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// h'(f, g) = f' + g'
	return args.front()->gradient(wrt) + args.back()->gradient(wrt);
}

Nodeptr mul_grad (std::vector<Nodeptr> args, Nodeptr& wrt)
{
	// h'(f, g) = f' * g + g' * f
	return args.front() * args.back()->gradient(wrt) +
		args.back() * args.front()->gradient(wrt);
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
	lhs = transpose(lhs, dim_swap({1, lhs->shape().n_rank() - 2}));
	lhs = ShapeTransform::get(lhs, Shape({
		Shape({(DimT) gdim, (DimT) fdim}), fshape}));
	// dh/dg = transpose_1,n_rank-2(
	//		id[shape:f1f0f...g1g1] * transpose(f)[f0f1f...])[shape:fg0g1])
	Nodeptr rhs = get_identity(gdim, f->type(), fshape) * f;
	rhs = transpose(rhs, dim_swap({0, rhs->shape().n_rank() - 1}));
	rhs = ShapeTransform::get(rhs, Shape({
		Shape({(DimT) gdim, (DimT) fdim}), gshape}));
	// todo: attempt to get rid of these checks
	if (f.get() == wrt.get() && g.get() == wrt.get())
	{
		return lhs + rhs;
	}
	else if (f.get() == wrt.get())
	{
		return lhs;
	}
	else if (g.get() == wrt.get())
	{
		return rhs;
	}
	// df/dx
	Nodeptr dlhs = f->gradient(wrt);
	if (false == higher_order(dlhs->shape()))
	{
		dlhs = group(dlhs);
	}
	// dg/dx
	Nodeptr drhs = g->gradient(wrt);
	if (false == higher_order(drhs->shape()))
	{
		drhs = group(drhs);
	}
	return matmul(dlhs, lhs) + matmul(drhs, rhs);
}

static EnumMap<OPCODE,Grader> graders =
{
	{ADD, add_grad},
	{MUL, mul_grad},
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
