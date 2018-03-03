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

static inline demuxer* dim_demuxer (const varptr a, size_t dimension, std::string label)
{
	return demuxer::get(a, 
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
		std::vector<size_t> coords = tensorshape(slist).coord_from_idx(idx);
		std::vector<size_t> out(n);
		for (size_t j = 0; j < n; ++j)
		{
			coords[dimension] = j;
			out[j] = inshape.flat_idx(coords);
		}
		return out;
	},
	[dimension](tensorshape inshape)
	{
		assert(inshape.rank() > dimension);
		size_t slimit = inshape.as_list()[dimension];
		return tensorshape(std::vector<size_t>{slimit});
	}, label);
}

static inline GLUE_F get_reduce_gluer (size_t dimension)
{
	return [dimension](VARR_T dest, CVAR_T src, unsigned short bytesize, size_t i)
	{
		char* cdest = (char*) dest.first;
		const char* csrc = (const char*) src.first;
		size_t srcn = src.second.n_elems();
		if (dest.second.rank() > 1)
		{
			std::vector<size_t> slist = dest.second.as_list();
			slist[dimension] = 1;
			std::vector<size_t> coords = tensorshape(slist).coord_from_idx(i);
			size_t desti = 0;
			for (size_t srci = 0; srci < srcn; ++srci)
			{
				coords[dimension] = srci;
				desti = dest.second.flat_idx(coords);
				memcpy(cdest + desti * bytesize, csrc + srci * bytesize, bytesize);
			}
		}
		else
		{
			memcpy(cdest + i * srcn * bytesize, csrc, srcn * bytesize);
		}
	};
}

static inline varptr reduce (const varptr a, size_t dimension, 
	std::string op, std::function<varptr(varptr)> reduce_op)
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
	GLUE_F gluef = get_reduce_gluer(dimension);
	return muxer::get({MUXPAIR{
		dim_demuxer(a, dimension, dmx_label), 
		gluef
	}},
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
	},
	[reduce_op](std::vector<std::vector<inode*> > sliceargs)
	{
		// assert sliceargs.size() == 1
		std::vector<inode*>& sarg = sliceargs.front();
		std::vector<varptr> slices;
		for (inode* varg : sarg)
		{
			slices.push_back(reduce_op(varg));
		}
		return slices;
	}, gluef, mux_label);
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
