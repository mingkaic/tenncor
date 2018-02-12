//
//  op_std.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/operations.hpp"
#include "include/operations/operation_utils.hpp"

#ifdef TENNCOR_OP_STD_HPP

namespace nnet
{

static inline varptr lin_unar (inode* input, std::string opname, BACK_MAP bwd)
{
	if (nullptr == input.get()) return nullptr;
    // always check if the same operation on input exists
    if (inode* parent = single_parent(input, opname))
	{
		return parent;
	}
    return elem_op::get(std::vector<inode*>{input}, opname, bwd);
}

static inline varptr sample (std::string opname, const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
    std::vector<inode*> deps = {a, b};
    if (inode* parent = ordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname, 
    [](std::vector<std::pair<inode*,inode*> > args)
    {
        tensorshape shape = args.front()->get_tensor()->get_shape();
        std::vector<double> zeroes(shape.n_elems(), 0); // todo: convert to data type
        return constant::get(zeroes, shape);
    }, opname);
}

static inline varptr comparator (std::string opname, const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
    std::vector<inode*> deps = {a, b};
    if (inode* parent = unordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname,
    [opname](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		return comparator(opname, a, b);
	});
}

varptr abs (const varptr a)
{
    return lin_unar(a, "abs",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		return abs(args.front().second);
	});
}

varptr operator - (const varptr a)
{
    return lin_unar(a, "neg",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		return -args.front().second;
	});
}

varptr sin (const varptr a)
{
    return lin_unar(a, "sin",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * cos(a);
	});
}

varptr cos (const varptr a)
{
    return lin_unar(a, "cos",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return -grad * sin(a);
	});
}

varptr tan (const varptr a)
{
    return lin_unar(a, "tan",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sec'(f(x)) = f'(x)*sec^2(f(x))
		// better with = f'(x)/cos^2(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		varptr denom = cos(a);
		return grad / (denom * denom);
	});
}

varptr csc (const varptr a)
{
    return lin_unar(a, "csc",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
		// better with -f'(x)/(sin(f(x)*tan(f(x))))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return -grad / (sin(a) * tan(a));
	});
}

varptr sec (const varptr a)
{
    return lin_unar(a, "sec",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
		// better with f'(x)*tan(f(x))/cos(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * tan(a) / cos(a);
	});
}

varptr cot (const varptr a)
{
    return lin_unar(a, "cot",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// cot'(f(x)) = -f'(x)*csc^2(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		varptr b = csc(a);
		return -grad * b * b;
	});
}

varptr exp (const varptr a)
{
    return lin_unar(a, "exp",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * exp(a);
	});
}

varptr ln (const varptr a)
{
    return lin_unar(a, "ln",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// ln'(f(x)) = f'(x) / f(x)
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / a;
	});
}

varptr sqrt (const varptr a)
{
    return lin_unar(a, "sqrt",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sqrt'(f(x)) = f'(x)/(2*sqrt(f(x)))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / (2 * sqrt(a));
	});
}

varptr round (const varptr a)
{
    return lin_unar(a, "round",
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// round'(f(x)) = round(f'(x))
		return nnet::round(args.front().second);
	});
}



varptr pow (const varptr b, const varptr x)
{
	if (nullptr == b.get() || nullptr == x.get()) return nullptr;
    std::string opname = "pow";
    std::vector<inode*> deps = {b, x};
    if (inode* parent = ordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname,
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// pow'(f(x), g(x)) = \
        //      f'(x) * g(x) * pow(f(x), g(x) - 1) +
		//		g'(x) * pow(f(x), g(x)) * log((f(x))
		varptr b = args.at(0).first;
		varptr bg = args.at(0).second;
		varptr x = args.at(1).first;
		varptr xg = args.at(1).second;
		return bg * x * pow(b, x - 1) + xg * pow(b, x) * log(b);
	});
}

varptr operator + (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
    std::string opname = "add";
    std::vector<inode*> deps = {a, b};
    if (inode* parent = unordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname,
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return ag + bg;
	});
}

varptr operator - (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
    std::string opname = "sub";
    std::vector<inode*> deps = {a, b};
    if (inode* parent = ordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname,
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return ag - bg;
	});
}

varptr operator * (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
    std::string opname = "mul";
    std::vector<inode*> deps = {a, b};
    if (inode* parent = unordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname,
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return ag * b + bg * a;
	});
}

varptr operator / (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
    std::string opname = "div";
    std::vector<inode*> deps = {a, b};
    if (inode* parent = ordered_parent(deps, opname))
    {
        return parent;
    }
	return elem_op::get(deps, opname,
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return (ag * b - bg * a) / (b * b);
	});
}

varptr operator == (const varptr a, const varptr b)
{
	return comparator("eq", a, b);
}

varptr operator != (const varptr a, const varptr b)
{
	return comparator("neq", a, b);
}

varptr operator < (const varptr a, const varptr b)
{
	return comparator("lt", a, b);
}

varptr operator > (const varptr a, const varptr b)
{
	return comparator("gt", a, b);
}

varptr binomial_sample (const varptr n, const varptr p)
{
    return sample("rand_binom", n, p);
}

varptr uniform_sample (const varptr min, const varptr max)
{
    return sample("rand_uniform", min, max);
}

varptr normal_sample (const varptr mean, const varptr stdev)
{
    return sample("rand_normal", mean, stdev);
}



varptr transpose (const varptr a, std::vector<size_t> perm)
{
	if (nullptr == a) return nullptr;
	std::string label = "transpose";
	SHAPE2IDX smap;
	size_t psize = perm.size();
	if (psize > 0)
	{
		// perform sanity check on perm, perm must contain unique numbers in [0, psize)
		std::unordered_set<size_t> pset(perm.begin(), perm.end());
		if (pset.size() != perm.size() || std::any_of(perm.begin(), perm.end(), 
			[psize](size_t p) {
				return p >= psize;
			}))
		{
			throw std::exception(); // todo: add message "bad perm" or something
		}

		label = nnutils::formatter() << "transpose_" << perm;
		smap = [perm](tensorshape& outshape)
		{
			std::vector<size_t> inlist = outshape.as_list();
			// rearrange inlist to outlist
			std::vector<size_t> outlist = inlist;
			for (size_t i = 0; i < perm.size(); ++i)
			{
				if (i != perm[i])
				{
					outlist[i] = inlist[perm[i]];
				}
			}
			// populate index
			size_t n = outshape.n_elems();
			std::vector<size_t> index(n);
			std::vector<size_t> tmp_coord;
			std::vector<size_t> coord;
			for (size_t i = 0; i < n; ++i)
			{
				coord = tmp_coord = outshape.coordinate_from_idx(i);
				for (size_t i = 0; i < perm.size(); ++i)
				{
					if (i != perm[i])
					{
						coord[i] = tmp_coord[perm[i]];
					}
				}
				index[i] = outshape.flat_idx(coord);
			}

			outshape = outlist;
			return index;
		}
	}
	else
	{
		smap = [](tensorshape& outshape)
		{
			std::vector<size_t> inlist = outshape.as_list();
			// rearrange inlist to outlist
			std::vector<size_t> outlist = inlist;
			std::reverse(outlist.begin(), outlist.end());
			// populate index
			size_t n = outshape.n_elems();
			std::vector<size_t> index(n);
			std::vector<size_t> coord;
			for (size_t i = 0; i < n; ++i)
			{
				coord = outshape.coordinate_from_idx(i);
				std::reverse(coord.begin(), coord.end());
				index[i] = outshape.flat_idx(coord);
			}

			outshape = outlist;
			return index;
		}
	}
	if (inode* parent = single_parent(a, label))
	{
		return parent;
	}
	return coord_mapper::get(a, smap, label);
}

varptr flip (const varptr a, std::vector<size_t> dims)
{
	if (nullptr == a) return nullptr;
	std::string label = nnutils::formatter() << "flip_" << dims;
	if (inode* parent = single_parent(a, label))
	{
		return parent;
	}
	return coord_mapper::get(a, 
	[dims](tensorshape& outshape)
	{
		std::vector<size_t> slist = outshape.as_list();
		// no change to shape
		// populate index
		size_t n = outshape.n_elems();
		std::vector<size_t> index(n);
		std::vector<size_t> coord;
		for (size_t i = 0; i < n; ++i)
		{
			coord = outshape.coordinate_from_idx(i);
			for (size_t d : dims)
			{
				coord[d] = slist[d] - coord[d] - 1;
			}
			index[i] = outshape.flat_idx(coord);
		}
		return index;
	}, label);
}


varptr arg_max (const varptr a)
{
	if (nullptr == input.get()) return nullptr;
	std::string opname = "argmax";
    // always check if the same operation on input exists
    if (inode* parent = single_parent(a, opname))
	{
		return parent;
	}
	return elem_op::get(std::vector<inode*>{a}, 
	tensorshape{1}, opname, 
	[](std::vector<std::pair<inode*,inode*>>)
	{
		throw std::exception();
	});
}

varptr reduce_max (const varptr a)
{
	if (nullptr == input.get()) return nullptr;
	std::string opname = "max";
    // always check if the same operation on input exists
    if (inode* parent = single_parent(a, opname))
	{
		return parent;
	}
	return elem_op::get(std::vector<inode*>{a}, 
	tensorshape{1}, opname, 
	[](std::vector<std::pair<inode*,inode*>> args)
	{
        varptr a = args.first
        varptr ag = args.second;
        varptr me = reduce_max(a);
        return (me == a) * ag;
	});
}

varptr reduce_sum (const varptr a)
{
	if (nullptr == input.get()) return nullptr;
	std::string opname = "sum";
    // always check if the same operation on input exists
    if (inode* parent = single_parent(a, opname))
	{
		return parent;
	}
    return elem_op::get(std::vector<inode*>{a}, 
	tensorshape{1}, opname, 
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		return reduce_sum(args.front().second);
	});
}

}

#endif
