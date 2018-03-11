//
//  op_comp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/operations.hpp"
#include "include/operations/operation_utils.hpp"

#include "include/graph/connector/muxer.hpp"
#include "include/graph/connector/functor.hpp"

#ifdef TENNCOR_OP_COM_HPP

namespace nnet
{

static inline varptr reduce (const varptr a, size_t dimension, 
	std::string op, std::function<varptr(varptr)> reduce_op)
{
	if (nullptr == a.get()) return nullptr;
	std::string dmx_label = nnutils::formatter() << "reduce_" << op << "_" << dimension;
	if (inode* parent = single_parent(a, dmx_label))
	{
		return parent;
	}
	// return functor::get({a},
	// [](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	// {
	// 	inode* arg = args[0];
	// },
	// [](inode* wrt, std::vector<inode*> args)
	// {
	// 	inode* arg = args[0];
	// 	arg->derive(wrt);
	// }, dmx_label);



	return muxer::get(a, dimension, reduce_op, dmx_label);
}

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


varptr reduce_max (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "max", [](varptr slice)
	{
		return reduce_max(slice);
	});
}

varptr reduce_sum (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "sum", [](varptr slice)
	{
		return reduce_sum(slice);
	});
}

varptr reduce_mean (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "mean", [](varptr slice)
	{
		return reduce_mean(slice);
	});
}

varptr reduce_l2norm (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "l2norm", [](varptr slice)
	{
		return reduce_l2norm(slice);
	});
}


varptr arg_max (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "arg_max", [](varptr slice)
	{
		return arg_max(slice);
	});
}

}

#endif
