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

static inline varptr reduce (const varptr a, size_t dimension, std::string op, SLICE_OP slice)
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
	[dimension](tensorshape& outshape)
	{
		size_t rank = outshape.rank();
		size_t nslices = 0;
		std::vector<size_t> slist = outshape.as_list();
		if (rank > dimension)
		{
			outshape = {slist[dimension]};
			slist[dimension] = 1;
			nslices = std::accumulate(slist.begin(), slist.end(), 
			(size_t) 1, std::multiplies<size_t>());
		}
		return nslices;
	},
	[dimension](tensorshape inshape, size_t idx)
	{
		assert(inshape.rank() > dimension);
		size_t slimit = inshape.as_list()[dimension];
		std::vector<size_t> coords = inshape.coordinate_from_idx(idx);
		if (dimension == coords.size())
		{
			coords.push_back(0);
		}
		else if (dimension == 0)
		{
			std::vector<size_t> temp = coords;
			coords = {0};
			coords.insert(coords.end(), temp.begin(), temp.end());
		}

		std::vector<size_t> out(slimit);
		for (size_t j = 0; j < slimit; ++j)
		{
			coords[dimension] = j;
			out[j] = inshape.flat_idx(coords);
		}
		return out;
	}, dmx_label);
	return muxer::get({dm}, 
    [dimension](std::vector<tensorshape> shapes)
    {
        std::vector<size_t> slist = shapes[0].as_list();
        if (1 == slist.size())
        {
            slist[0] = 1;
        }
        else if (0 == dimension)
        // pop front
        {
            slist = std::vector<size_t>(slist.begin() + 1, slist.end());
        }
        else if (slist.size() - 1 == dimension)
        {
            slist.pop_back();
        }
        else
        {
            slist[dimension] = 1;
        }
        return tensorshape(slist);
    }, slice,
    [](SHARED_VARR dest, const SHARED_VARR src, 
        signed short bytesize, size_t i)
    {
        assert(src.second.n_elems() == 1 && dest.second.n_elems() > i);
        char* cdest = (char*) dest.first.get();
        char* csrc = (char*) src.first.get();
        std::memcpy(cdest + i * bytesize, csrc, bytesize);
    }, dmx_label);
}

varptr clip (const varptr a, const varptr min, const varptr max)
{
	if (nullptr == a.get() || nullptr == min.get() || nullptr == max.get()) return nullptr;
    varptr lt_min = a < min;
    varptr gt_max = a > max;
    return (!lt_min & !gt_max) * a + lt_min * min + gt_max * max;
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
        n = shape_dep::get(a, [](tensorshape& outshape)
        {
            return std::vector<size_t>{outshape.n_elems()};
        }, std::vector<size_t>{1}, "n_elems");
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

