//
//  matmul_grad.cpp
//  kiln
//

#include <cassert>

#include "kiln/matmul_grad.hpp"
#include "kiln/operators.hpp"

#ifdef KILN_MATMUL_GRAD_HPP

namespace kiln
{

static Identifier* tracify (Identifier* arg)
{
	clay::State state = arg->get_state();
	clay::Shape& shape = state.shape_;
	std::vector<size_t> slist = shape.as_list();
	if (slist.size() > 1)
	{
		slist[0] *= slist[1];
		slist.erase(slist.begin() + 1);
	}
	Identifier* flat = reshape(arg, std::vector<uint64_t>(
		slist.begin(), slist.end()));
	return trace_expand(flat, (uint64_t) 0);
}

Identifier* matmul_grad (Identifier*, GradArgsT args)
{
	// matmul'(f, g) = matmul(f', dmatmul/df) + matmul(g', dmatmul/dg)
	// dmatmul/df = jacobian(g), dmatmul/dg = jacobian(f)
	// C = matmul(A, B), where A has shape <m, n>, B has shape <p, m>
	// C has shape <p, n>
	// c_xy = sum_k=0,m-1(a_ky * b_xk)
	// d c_xy / d a_vu = sum(b_xk yf a_ky == a_vu else 0)
	// d c_xy / d a_vy = b_xv
	// d c_xy / d a_uk = 0 for k != y

	Identifier* a = args.front().first.arg_;
	Identifier* b = args.back().first.arg_;
	Identifier* da = args.front().second.arg_;
	Identifier* db = args.back().second.arg_;
	// process both arguments as jacobians
	Identifier* lhs = nullptr;
	Identifier* rhs = nullptr;
	if (nullptr != da)
	{
		clay::Shape dashape = da->get()->get_shape();
		clay::Shape ashape = a->get()->get_shape();
		assert(ashape.is_fully_defined());
		if (dashape.at(0) != ashape.n_elems())
		// this is an unreliable way of detecting non-jacobian da
		// todo: improve this check
		{
			da = tracify(da);
		}
		Identifier* jb = jacobian(a, b, 0, 1);
		lhs = matmul(da, jb);
	}
	if (nullptr != db)
	{
		clay::Shape dbshape = db->get()->get_shape();
		clay::Shape bshape = b->get()->get_shape();
		assert(bshape.is_fully_defined());
		if (dbshape.at(0) != bshape.n_elems())
		{
			db = tracify(db);
		}
		Identifier* ja = jacobian(a, b, 1, 0);
		rhs = matmul(db, ja);
	}
	if (nullptr == lhs)
	{
		return rhs;
	}
	else if (nullptr == rhs)
	{
		return lhs;
	}
	return add(lhs, rhs);
}

}

#endif
