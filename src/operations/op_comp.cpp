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
	}, label);
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
	return muxer::get({dim_demuxer(a, dimension, dmx_label)}, 
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
	},
	[dimension](VARR_T dest, CVAR_T src, unsigned short bytesize, size_t i)
	{
		char* cdest = (char*) dest.first;
		const char* csrc = (const char*) src.first;
		size_t srcn = src.second.n_elems();
		if (dest.second.rank() > 1)
		{
			std::vector<size_t> slist = dest.second.as_list();
			slist[dimension] = 1;
			std::vector<size_t> coords = tensorshape(slist).coordinate_from_idx(i);
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
	}, mux_label);
}

static inline tensorshape matmul_shaper (std::vector<tensorshape> shapes)
{
	tensorshape& t1s = shapes[0];
	tensorshape& t2s = shapes[1];

	std::vector<size_t> al = t1s.as_list();
	std::vector<size_t> bl = t2s.as_list();
	size_t rank1 = t1s.rank();
	size_t rank2 = t2s.rank();

	// account for vectors
	size_t ax = rank1 ? al[0] : 0;
	size_t ay = rank1> 1 ? al[1] : 1;
	size_t bx = rank2 ? bl[0] : 0;
	size_t by = rank2> 1 ? bl[1] : 1;

	// ensure the dimensions beyond 2d are equal
	size_t minend = std::min(rank1, rank2);
	std::vector<size_t> beyond2d;
	if (minend> 2)
	{
		auto ait = al.begin()+2;
		auto aet = al.begin()+minend;
		if (std::equal(ait, aet, bl.begin()+2))
		{
			beyond2d.insert(beyond2d.end(), ait, aet);
		}
		else
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
		// check that remaining shape values are ones,
		// otherwise one shape is larger than the other
		auto it = rank1> rank2 ? al.begin() : bl.begin();
		auto et = rank1> rank2 ? al.end() : bl.end();
		if (!std::all_of(it + minend, et, [](size_t e) { return e == 1; }))
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple different sized shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
	}

	// get resulting shape
	std::vector<size_t> res_shape;
	if (ax == by)
	{
		res_shape = {bx, ay};
	}
	else
	{
		std::stringstream ss;
		ss << "matmul shapes ";
		print_shape(t1s, ss);
		ss << "and ";
		print_shape(t2s, ss);
		ss << "do not match";
		throw std::logic_error(ss.str());
	}
	res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
	return res_shape;
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


varptr matmul (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	std::string dlabel = "demux_matmul";
	std::string mlabel = "mux_matmul";
	inode* parenta = single_parent(a, dlabel);
	inode* parentb = single_parent(b, dlabel);
	if (parenta && parentb)
	{
		if (inode* true_parent = ordered_parent({parenta, parentb}, mlabel))
		{
			return true_parent;
		}
	}
	// demux a by rows and b by cols
	demuxer* dma;
	demuxer* dmb;
	if (parenta)
	{
		dma = static_cast<demuxer*>(parenta);
	}
	else
	{
		dma = dim_demuxer(a, 0, dlabel);
	}
	if (parentb)
	{
		dmb = static_cast<demuxer*>(parentb);
	}
	else
	{
		dmb = dim_demuxer(b, 1, dlabel);
	}
	return muxer::get({dma, dmb}, matmul_shaper, 
	[](std::vector<std::vector<inode*> > sliceargs)
	{
		// assert sliceargs.size() == 2
		std::vector<inode*>& aslices = sliceargs.front();
		std::vector<inode*>& bslices = sliceargs.back();
		size_t nrows = aslices.size();
		size_t ncols = bslices.size();
		std::vector<varptr> slices(nrows * ncols);
		// i = y * ncols + x
		// slices[i] = reduce_sum(aslices[y] * bslices[x])
		for (size_t y = 0; y < nrows; ++y)
		{
			for (size_t x = 0; x < ncols; ++x)
			{
				slices[y * ncols + x] = reduce_sum(varptr(aslices[y]) * varptr(bslices[x]));
			}
		}
		return slices;
	},
	[](VARR_T dest, CVAR_T src, unsigned short bytesize, size_t i)
	{
		// glue everything together
		// asssert src.second.n_elems() == 1;
		char* cdest = (char*) dest.first;
		const char* csrc = (const char*) src.first;
		memcpy(cdest + i * bytesize, csrc, bytesize);
	}, mlabel);
}

}

#endif
