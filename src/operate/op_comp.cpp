//
//  op_comp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/operations.hpp"
#include "include/operate/operation_utils.hpp"

#include "include/operate/common.hpp"

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
	assert_shape(cap, std::vector<size_t>{1});
	varptr l2 = reduce_l2norm(a);
	varptr is_clip = l2 > cap;
	return is_clip * a * cap / l2 + !is_clip * a;
}



varptr reduce_mean (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	return reduce_sum(a) / n_elems(a);
}

varptr reduce_l2norm (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	return sqrt(reduce_sum(a * a)); 
}



varptr arg_max (const varptr a, uint64_t dimension)
{
	return arg_max(a, constant::get<uint64_t>(dimension));
}

varptr arg_max (const varptr a, varptr dimension)
{
	if (nullptr == a.get()) return nullptr;
	if (nullptr == dimension.get())
	{
		dimension = constant::get<uint64_t>(0);
	}
	assert_shape(dimension, std::vector<size_t>{1});
	assert_type(dimension, nnet::UINT64);
	// always check if the same operation on input exists
	OPCODE op = ARGMAX;
	if (inode* parent = ordered_parent({a, dimension}, op))
	{
		return parent;
	}
	return arg_func(a, dimension, "argmax", op,
	[](std::vector<std::pair<inode*,varptr> >) -> varptr
	{
		throw std::exception();
	});
}

varptr reduce_max (const varptr a, uint64_t dimension)
{
	return reduce_max(a, constant::get<uint64_t>(dimension));
}

varptr reduce_max (const varptr a, varptr dimension)
{
	if (nullptr == a.get()) return nullptr;
	if (nullptr == dimension.get())
	{
		dimension = constant::get<uint64_t>(0);
	}
	assert_shape(dimension, std::vector<size_t>{1});
	assert_type(dimension, nnet::UINT64);
	// always check if the same operation on input exists
	OPCODE op = RMAX;
	if (inode* parent = ordered_parent({a, dimension}, op))
	{
		return parent;
	}
	return reduce_func(a, dimension, "max", op,
	[](std::vector<std::pair<inode*,varptr> > args) -> varptr
	{
		varptr a = args.front().first;
		varptr ag = args.front().second;
		inode* c = args.back().first;
		uint64_t dimension = expose<uint64_t>(c)[0];
		varptr me = reduce_max(a, dimension);
		varptr bitmap = expand(me, n_dimension(a, dimension), dimension);
		return (bitmap == a) * ag;
	});
}

varptr reduce_sum (const varptr a, uint64_t dimension)
{
	return reduce_sum(a, constant::get<uint64_t>(dimension));
}

varptr reduce_sum (const varptr a, varptr dimension)
{
	if (nullptr == a.get()) return nullptr;
	if (nullptr == dimension.get())
	{
		dimension = constant::get<uint64_t>(0);
	}
	assert_shape(dimension, std::vector<size_t>{1});
	assert_type(dimension, nnet::UINT64);
	OPCODE op = RSUM;
	// always check if the same operation on input exists
	if (inode* parent = ordered_parent({a, dimension}, op))
	{
		return parent;
	}
	return reduce_func(a, dimension, "sum", op,
	[](std::vector<std::pair<inode*,varptr> > args) -> varptr
	{
		return args.front().second;
	});
}

varptr reduce_mean (const varptr a, uint64_t dimension)
{
	if (nullptr == a.get()) return nullptr;
	return reduce_sum(a, dimension) / n_dimension(a, dimension);
}

varptr reduce_mean (const varptr a, const varptr dimension)
{
	if (nullptr == a.get()) return nullptr;
	return reduce_sum(a, dimension) / n_dimension(a, dimension);
}

varptr reduce_l2norm (const varptr a, uint64_t dimension)
{
	if (nullptr == a.get()) return nullptr;
	return sqrt(reduce_sum(a * a, dimension));
}

varptr reduce_l2norm (const varptr a, const varptr dimension)
{
	if (nullptr == a.get()) return nullptr;
	return sqrt(reduce_sum(a * a, dimension));
}

}

#endif
