//
//  op_comp.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/operations.hpp"
#include "include/operations/operation_utils.hpp"

#ifdef TENNCOR_OP_COM_HPP

namespace nnet
{

static inline varptr reduce (const varptr a, size_t dimension, std::string op, VAROP_F reduce_op)
{
	if (nullptr == a.get()) return nullptr;
	std::string dmx_label = nnutils::formatter() << "reduce_" << op << "_" << dimension;
	std::string mux_label = "reduce_muxer";
	if (inode* parent = single_parent(a, dmx_label))
	{
		inode* true_parent = single_parent(parent, mux_label);
		assert(nullptr != true_parent);
		return true_parent;
	}
	demuxer* dm = demuxer::get(a, 
	[dimension](tensorshape shape)
	{
		size_t rank = shape.rank();
		size_t nslices = 0;
		std::vector<size_t> slist = shape.as_list();
		if (rank > dimension)
		{
			slist[dimension] = 1;
			nslices = std::accumulate(slist.begin(), slist.end(), 
				(size_t) 1, std::multiplies<size_t>());
		}
		return nslices;
	},
	[dimension](tensorshape outshape, const tensorshape inshape, size_t idx)
	{
		size_t n = outshape.n_elems();
		size_t rank = inshape.rank();
		assert(rank > dimension);
		std::vector<size_t> slist = inshape.as_list();
		slist[dimension] = 1;
		std::vector<size_t> coords = tensorshape(slist).coordinate_from_idx(idx);
		std::vector<size_t> out(n);
		for (size_t j = 0; j < n; ++j)
		{
			coords[dimension] = j;
			out[j] = inshape.flat_idx(coords);
		}
		return out;
	}, 
	[dimension](tensorshape shape)
	{
		assert(shape.rank() > dimension);
		size_t slimit = shape.as_list()[dimension];
		return tensorshape(std::vector<size_t>{slimit});
	}, dmx_label);
	return muxer::get({dm}, 
	[dimension](std::vector<tensorshape> shapes)
	{
		std::vector<size_t> slist = shapes[0].as_list();
		if (1 == slist.size())
		{
			slist[0] = 1;
		}
		else
		{
			slist.erase(slist.begin() + dimension);
		}
		return tensorshape(slist);
	}, reduce_op,
	[](VARR_T dest, CVAR_T src, unsigned short bytesize, size_t i)
	{
		size_t srcn = src.second.n_elems();
		char* cdest = (char*) dest.first;
		const char* csrc = (const char*) src.first;
		std::memcpy(cdest + i * srcn * bytesize, csrc, srcn * bytesize);
	}, dmx_label);
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
	if (nullptr == a.get()) return nullptr;
	varptr n = single_parent(a, "n_elems");
	if (nullptr == n.get())
	{
		n = shape_dep::get(a,
		[](tensorshape inshape)
		{
			return std::vector<size_t>{inshape.n_elems()};
		},
		[](tensorshape)
		{
			return tensorshape(std::vector<size_t>{1});
		}, "n_elems");
	}
	return reduce_sum(a) / n;
}

varptr reduce_l2norm (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	return sqrt(reduce_sum(pow(a, 2)));
}


varptr reduce_max (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "max", [](std::vector<varptr> slices)
	{
		// assert(slices.size() == 1);
		return reduce_max(slices[0]);
	});
}

varptr reduce_sum (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "sum", [](std::vector<varptr> slices)
	{
		// assert(slices.size() == 1);
		return reduce_sum(slices[0]);
	});
}

varptr reduce_mean (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "mean", [](std::vector<varptr> slices)
	{
		// assert(slices.size() == 1);
		return reduce_mean(slices[0]);
	});
}

varptr reduce_l2norm (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "l2norm", [](std::vector<varptr> slices)
	{
		// assert(slices.size() == 1);
		return reduce_l2norm(slices[0]);
	});
}


varptr arg_max (const varptr a, size_t dimension)
{
	return reduce(a, dimension, "arg_max", [](std::vector<varptr> slices)
	{
		// assert(slices.size() == 1);
		return arg_max(slices[0]);
	});
}

}

#endif
