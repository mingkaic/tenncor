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

static inline varptr lin_unar (std::string opname, inode* input, BACKMAP_F bwd)
{
	if (nullptr == input) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(input, opname))
	{
		return parent;
	}
	return elem_op::get(std::vector<inode*>{input}, opname, bwd);
}

static inline varptr sample (std::string opname, inode* a, inode* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	std::vector<inode*> deps = {a, b};
	if (inode* parent = ordered_parent(deps, opname))
	{
		return parent;
	}
	return elem_op::get(deps, opname, 
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		tensorshape shape = args.front().first->get_tensor()->get_shape();
		std::vector<double> zeroes(shape.n_elems(), 0); // todo: convert to data type
		return constant::get(zeroes, shape);
	});
}

static inline varptr comparator (std::string opname, inode* a, inode* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	std::vector<inode*> deps = {a, b};
	if (inode* parent = unordered_parent(deps, opname))
	{
		return parent;
	}
	return elem_op::get(deps, opname,
	[opname](std::vector<std::pair<inode*,inode*> > args)
	{
		varptr a = args.front().first;
		varptr b = args.back().first;
		return comparator(opname, a, b);
	});
}

static inline varptr aggregate (std::string opname, inode* a, BACKMAP_F bwd)
{
	if (nullptr == a) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, opname))
	{
		return parent;
	}
	return elem_op::get(std::vector<inode*>{a}, 
	std::vector<size_t>{1}, opname, bwd);
}

varptr abs (const varptr a)
{
	return lin_unar("abs", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		return abs(args.front().second);
	});
}

varptr operator - (const varptr a)
{
	return lin_unar("neg", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		varptr ag = args.front().second;
		return -ag;
	});
}

varptr operator ! (const varptr a)
{
	return lin_unar("logic_not", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		varptr a = args.front().first;
		return !a;
	});
}

varptr sin (const varptr a)
{
	return lin_unar("sin", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * cos(a);
	});
}

varptr cos (const varptr a)
{
	return lin_unar("cos", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return -grad * sin(a);
	});
}

varptr tan (const varptr a)
{
	return lin_unar("tan", a,
	[](std::vector<std::pair<inode*,inode*> > args)
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
	return lin_unar("csc", a,
	[](std::vector<std::pair<inode*,inode*> > args)
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
	return lin_unar("sec", a,
	[](std::vector<std::pair<inode*,inode*> > args)
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
	return lin_unar("cot", a,
	[](std::vector<std::pair<inode*,inode*> > args)
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
	return lin_unar("exp", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * exp(a);
	});
}

varptr ln (const varptr a)
{
	return lin_unar("ln", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// ln'(f(x)) = f'(x) / f(x)
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / a;
	});
}

varptr sqrt (const varptr a)
{
	return lin_unar("sqrt", a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// sqrt'(f(x)) = f'(x)/(2*sqrt(f(x)))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / ((double) 2 * sqrt(a)); // todo: convert 2 to type
	});
}

varptr round (const varptr a)
{
	return lin_unar("round", a,
	[](std::vector<std::pair<inode*,inode*> > args)
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
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// pow'(f(x), g(x)) = \
		//		f'(x) * g(x) * pow(f(x), g(x) - 1) +
		//		g'(x) * pow(f(x), g(x)) * log((f(x))
		varptr b = args.front().first;
		varptr bg = args.front().second;
		varptr x = args.back().first;
		varptr xg = args.back().second;
		return bg * x * pow(b, x - 1) + xg * pow(b, x) * ln(b);
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
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		varptr ag = args.front().second;
		varptr bg = args.back().second;
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
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		varptr ag = args.front().second;
		varptr bg = args.back().second;
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
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr a = args.front().first;
		varptr ag = args.front().second;
		varptr b = args.back().first;
		varptr bg = args.back().second;
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
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// h'(f(x), g(x)) = (f'(x) * g(x) - f(x) * g'(x)) / g^2(x)
		//		= f'(x) / g(x) - (f(x) * g'(x)) / g(x)) / g(x)
		varptr a = args.front().first;
		varptr ag = args.front().second;
		varptr b = args.back().first;
		varptr bg = args.back().second;
		varptr second = a * bg / b;
		return (ag - second) / b;
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
	if (nullptr == a.get()) return nullptr;
	std::string label = "transpose";
	SIDX_F smap;
	USHAPE_F shaper;
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
		smap = [perm](tensorshape outshape, const tensorshape inshape)
		{
			// populate index
			size_t n = outshape.n_elems(); // inshape size if same as output
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
				index[i] = inshape.flat_idx(coord);
			}
			return index;
		};

		shaper = [perm](tensorshape inshape)
		{
			// rearrange inlist to outlist
			std::vector<size_t> inlist = inshape.as_list();
			std::vector<size_t> outlist = inlist;
			for (size_t i = 0; i < perm.size(); ++i)
			{
				if (i != perm[i])
				{
					outlist[i] = inlist[perm[i]];
				}
			}
			return tensorshape(outlist);
		};
	}
	else
	{
		smap = [](tensorshape outshape, const tensorshape inshape)
		{
			// populate index
			size_t n = outshape.n_elems();
			std::vector<size_t> index(n);
			std::vector<size_t> coord;
			for (size_t i = 0; i < n; ++i)
			{
				coord = outshape.coordinate_from_idx(i);
				std::reverse(coord.begin(), coord.end());
				index[i] = inshape.flat_idx(coord);
			}
			return index;
		};

		shaper = [](tensorshape inshape)
		{
			// rearrange inlist to outlist
			std::vector<size_t> slist = inshape.as_list();
			std::reverse(slist.begin(), slist.end());
			return tensorshape(slist);
		};
	}

	if (inode* parent = single_parent(a, label))
	{
		return parent;
	}
	return coord_mapper::get(a, smap, shaper, label);
}

varptr flip (const varptr a, std::vector<size_t> dims)
{
	if (nullptr == a.get()) return nullptr;
	std::string label = nnutils::formatter() << "flip_" << dims;
	if (inode* parent = single_parent(a, label))
	{
		return parent;
	}
	return coord_mapper::get(a, 
	[dims](tensorshape outshape, const tensorshape)
	{
		// assert inshape.is_compatible_with(outshape)
		std::vector<size_t> slist = outshape.as_list();
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
	}, [](tensorshape inshape) { return inshape; }, label);
}


varptr arg_max (const varptr a)
{
	return aggregate("argmax", a,
	[](std::vector<std::pair<inode*,inode*> >) -> varptr
	{
		throw std::exception();
	});
}

varptr reduce_max (const varptr a)
{
	return aggregate("max", a,
	[](std::vector<std::pair<inode*,inode*> > args) -> varptr
	{
		varptr a = args.front().first;
		varptr ag = args.front().second;
		varptr me = reduce_max(a);
		return (me == a) * ag;
	});
}

varptr reduce_sum (const varptr a)
{
	return aggregate("sum", a,
	[](std::vector<std::pair<inode*,inode*> > args) -> varptr
	{
		return reduce_sum(args.front().second);
	});
}

}

#endif
