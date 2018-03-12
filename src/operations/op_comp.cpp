//
//  op_comp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/operations.hpp"
#include "include/operations/operation_utils.hpp"

#include "include/graph/connector/agg_func.hpp"
#include "include/graph/connector/functor.hpp"

#ifdef TENNCOR_OP_COM_HPP

namespace nnet
{

varptr clip (const varptr a, const varptr min, const varptr max)
{
	if (nullptr == a.get() || nullptr == min.get() || nullptr == max.get()) return nullptr;
	varptr lt_min = a < min;
	varptr gt_max = a > max;
	return !lt_min * !gt_max * a + lt_min * min + gt_max * max;
}

//! normalize clip values with capacity cap
varptr clip_norm (const varptr a, const varptr cap)
{
	if (nullptr == a.get() || nullptr == cap.get()) return nullptr;
	varptr l2 = reduce_l2norm(a);
	varptr is_clip = l2 > cap;
	return is_clip * a * cap / l2 + !is_clip * a;
}


varptr reduce_mean (const varptr a)
{
	return reduce_sum(a) / n_elems(a);
}

varptr reduce_l2norm (const varptr a)
{
	return sqrt(reduce_sum(a * a)); 
}


varptr arg_max (const varptr a, size_t dimension)
{
	if (nullptr == a.get()) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, nnutils::formatter() << "argmax_" << dimension))
	{
		return parent;
	}
	return agg_func(a, "argmax", dimension,
	[](std::vector<std::pair<inode*, inode*> >) -> varptr
	{
		throw std::exception();
	});
}


varptr reduce_max (const varptr a, size_t dimension)
{
	if (nullptr == a.get()) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, nnutils::formatter() << "max_" << dimension))
	{
		return parent;
	}
	return agg_func(a, "max", dimension,
	[dimension](std::vector<std::pair<inode*, inode*> > args) -> varptr
	{
		varptr a = args.front().first;
		varptr ag = args.front().second;
		varptr me = reduce_max(a, dimension);
		varptr bitmap = expand(me, n_dimension(a, dimension), dimension);
		return (bitmap == a) * ag;
	});
}

varptr reduce_sum (const varptr a, size_t dimension)
{
	if (nullptr == a.get()) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, nnutils::formatter() << "sum_" << dimension))
	{
		return parent;
	}
	return agg_func(a, "sum", dimension,
	[](std::vector<std::pair<inode*, inode*> > args) -> varptr
	{
		return args.front().second;
	});
}

varptr reduce_mean (const varptr a, size_t dimension)
{
	return reduce_sum(a, dimension) / n_dimension(a, dimension);
}

varptr reduce_l2norm (const varptr a, size_t dimension)
{
	return sqrt(reduce_sum(a * a, dimension));
}

}

#endif
