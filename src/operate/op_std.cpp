//
//  op_std.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-14.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/operations.hpp"
#include "include/operate/operation_utils.hpp"

#include "include/graph/func/elem_func.hpp"
#include "include/graph/func/shape_func.hpp"
#include "include/graph/func/coord_func.hpp"
#include "include/graph/func/agg_func.hpp"

#ifdef TENNCOR_OP_STD_HPP

namespace nnet
{

inode* run_opcode (std::vector<inode*> args, OPCODE code)
{
	switch (code)
	{
		case ABS:
			return abs(varptr(args[0]));
		case NEG:
			return -(varptr(args[0]));
		case NOT:
			return !(varptr(args[0]));
		case SIN:
			return sin(varptr(args[0]));
		case COS:
			return cos(varptr(args[0]));
		case TAN:
			return tan(varptr(args[0]));
		case EXP:
			return exp(varptr(args[0]));
		case LOG:
			return log(varptr(args[0]));
		case SQRT:
			return sqrt(varptr(args[0]));
		case ROUND:
			return round(varptr(args[0]));
		case POW:
			return pow(varptr(args[0]), varptr(args[1]));
		case ADD:
			return varptr(args[0]) + varptr(args[1]);
		case SUB:
			return varptr(args[0]) - varptr(args[1]);
		case MUL:
			return varptr(args[0]) * varptr(args[1]);
		case DIV:
			return varptr(args[0]) / varptr(args[1]);
		case EQ:
			return varptr(args[0]) == varptr(args[1]);
		case NE:
			return varptr(args[0]) != varptr(args[1]);
		case LT:
			return varptr(args[0]) < varptr(args[1]);
		case GT:
			return varptr(args[0]) > varptr(args[1]);
		case BINO:
			return binomial_sample(varptr(args[0]), varptr(args[1]));
		case UNIF:
			return uniform_sample(varptr(args[0]), varptr(args[1]));
		case NORM:
			return normal_sample(varptr(args[0]), varptr(args[1]));
		case TRANSPOSE:
			return transpose(varptr(args[0]), varptr(args[1]));
		case FLIP:
			return flip(varptr(args[0]), varptr(args[1]));
		case ARGMAX:
			if (args.size() > 1)
			{
				return arg_max(varptr(args[0]), varptr(args[1]));
			}
			return arg_max(varptr(args[0]));
		case RMAX:
			if (args.size() > 1)
			{
				return reduce_max(varptr(args[0]), varptr(args[1]));
			}
			return reduce_max(varptr(args[0]));
		case RSUM:
			if (args.size() > 1)
			{
				return reduce_sum(varptr(args[0]), varptr(args[1]));
			}
			return reduce_sum(varptr(args[0]));
		case EXPAND:
			return expand(varptr(args[0]), varptr(args[1]), varptr(args[2]));
		case N_ELEMS:
			return n_elems(varptr(args[0]));
		case N_DIMS:
			return n_dimension(varptr(args[0]), varptr(args[1]));
		case MATMUL:
			return matmul(varptr(args[0]), varptr(args[1]));
		default:
			break;
	}
	throw std::bad_function_call();
}

static inline varptr lin_unar (std::string opname, OPCODE op, inode* input, BACKMAP_F bwd)
{
	if (nullptr == input) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(input, opname))
	{
		return parent;
	}
	return elem_func(std::vector<inode*>{input}, opname, op, bwd);
}

static inline varptr sample (std::string opname, OPCODE op, inode* a, inode* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	std::vector<inode*> deps = {a, b};
	if (inode* parent = ordered_parent(deps, opname))
	{
		return parent;
	}
	return elem_func(deps, opname, op,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		tensorshape shape = args.front().first->get_tensor()->get_shape();
		std::vector<double> zeroes(shape.n_elems(), 0); // todo: convert to data type
		return constant::get<double>(zeroes, shape);
	});
}

static inline varptr comparator (std::string opname, OPCODE op, inode* a, inode* b)
{
	varptr out;
	if (nullptr == a && nullptr == b)
	{
		out = constant::get<double>((double) 1);
	}
	else if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	std::vector<inode*> deps = {a, b};
	if (inode* parent = unordered_parent(deps, opname))
	{
		return parent;
	}
	else
	{
		out = elem_func(deps, opname, op,
		[opname, op](std::vector<std::pair<inode*,inode*> > args)
		{
			varptr a = args.front().first;
			varptr b = args.back().first;
			return comparator(opname, op, a, b);
		});
	}
	return out;
}

varptr abs (const varptr a)
{
	return lin_unar("abs", ABS, a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		return abs(args.front().second);
	});
}

varptr operator - (const varptr a)
{
	return lin_unar("neg", NEG, a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		varptr ag = args.front().second;
		return -ag;
	});
}

varptr operator ! (const varptr a)
{
	return lin_unar("logic_not", NOT, a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		varptr a = args.front().first;
		return !a;
	});
}

varptr sin (const varptr a)
{
	return lin_unar("sin", SIN, a,
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
	return lin_unar("cos", COS, a,
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
	return lin_unar("tan", TAN, a,
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
	return (double) 1.0 / sin(a);
}

varptr sec (const varptr a)
{
	return (double) 1.0 / cos(a);
}

varptr cot (const varptr a)
{
	return (double) 1.0 / tan(a);
}

varptr exp (const varptr a)
{
	return lin_unar("exp", EXP, a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * exp(a);
	});
}

varptr log (const varptr a)
{
	return lin_unar("log", LOG, a,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// log'(f(x)) = f'(x) / f(x)
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / a;
	});
}

varptr sqrt (const varptr a)
{
	return lin_unar("sqrt", SQRT, a,
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
	return lin_unar("round", ROUND, a,
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
	return elem_func(deps, opname, POW,
	[](std::vector<std::pair<inode*,inode*> > args)
	{
		// pow'(f(x), g(x)) = \
		//		f'(x) * g(x) * pow(f(x), g(x) - 1) +
		//		g'(x) * pow(f(x), g(x)) * log((f(x))
		varptr b = args.front().first;
		varptr bg = args.front().second;
		varptr x = args.back().first;
		varptr xg = args.back().second;
		return bg * x * pow(b, x - 1) + xg * pow(b, x) * log(b);
	});
}

varptr operator + (const varptr a, const varptr b)
{
	if (nullptr == a.get()) return b;
	if (nullptr == b.get()) return a;
	std::string opname = "add";
	std::vector<inode*> deps = {a, b};
	if (inode* parent = unordered_parent(deps, opname))
	{
		return parent;
	}
	return elem_func(deps, opname, ADD,
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
	if (nullptr == a.get()) return -b;
	if (nullptr == b.get()) return a;
	std::string opname = "sub";
	std::vector<inode*> deps = {a, b};
	if (inode* parent = ordered_parent(deps, opname))
	{
		return parent;
	}
	return elem_func(deps, opname, SUB,
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
	return elem_func(deps, opname, MUL,
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
	if (nullptr == a.get()) return nullptr;
	if (nullptr == b.get()) throw std::exception(); // todo: divide by zero error
	std::string opname = "div";
	std::vector<inode*> deps = {a, b};
	if (inode* parent = ordered_parent(deps, opname))
	{
		return parent;
	}
	return elem_func(deps, opname, DIV,
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
	return comparator("eq", EQ, a, b);
}

varptr operator != (const varptr a, const varptr b)
{
	return comparator("neq", NE, a, b);
}

varptr operator < (const varptr a, const varptr b)
{
	return comparator("lt", LT, a, b);
}

varptr operator > (const varptr a, const varptr b)
{
	return comparator("gt", GT, a, b);
}

varptr binomial_sample (const varptr n, const varptr p)
{
	return sample("rand_binom", BINO, n, p);
}

varptr uniform_sample (const varptr min, const varptr max)
{
	return sample("rand_uniform", UNIF, min, max);
}

varptr normal_sample (const varptr mean, const varptr stdev)
{
	return sample("rand_normal", NORM, mean, stdev);
}



varptr transpose (const varptr a, std::vector<size_t> perm)
{
	varptr pvar;
	size_t psize = perm.size();
	if (psize > 0)
	{
		// perform sanity check on perm, perm must contain unique numbers in [0, psize)
		std::unordered_set<size_t> pset(perm.begin(), perm.end());
		if (pset.size() != perm.size() || std::any_of(perm.begin(), perm.end(), 
			[psize](size_t p) { return p >= psize; }))
		{
			throw std::exception(); // todo: add message "bad perm" or something
		}

		pvar = constant::get<uint64_t>(std::vector<uint64_t>(perm.begin(), perm.end()), 
			tensorshape(std::vector<size_t>{psize}));
	}
	return transpose(a, pvar);
}

varptr transpose (const varptr a, const varptr perm)
{
	if (nullptr == a.get()) return nullptr;
	SIDX_F smap;
	USHAPE_F shaper;
	std::vector<inode*> args = {a};
	if (nullptr == perm.get())
	{
		smap = [](tensorshape outshape, const tensorshape inshape, std::vector<uint64_t>)
		{
			// populate index
			size_t n = outshape.n_elems();
			std::vector<size_t> index(n);
			std::vector<size_t> coord;
			for (size_t i = 0; i < n; ++i)
			{
				coord = outshape.coord_from_idx(i);
				std::reverse(coord.begin(), coord.end());
				index[i] = inshape.flat_idx(coord);
			}
			return index;
		};

		shaper = [](tensorshape inshape, std::vector<uint64_t>)
		{
			// rearrange inlist to outlist
			std::vector<size_t> slist = inshape.as_list();
			std::reverse(slist.begin(), slist.end());
			return tensorshape(slist);
		};
	}
	else
	{
		args.push_back(perm);
		smap = [](tensorshape outshape, const tensorshape inshape, std::vector<uint64_t> perm)
		{
			// populate index
			size_t n = outshape.n_elems(); // inshape size if same as output
			std::vector<size_t> index(n);
			std::vector<size_t> tmp_coord;
			std::vector<size_t> coord;
			for (size_t i = 0; i < n; ++i)
			{
				coord = tmp_coord = outshape.coord_from_idx(i);
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

		shaper = [](tensorshape inshape, std::vector<uint64_t> perm)
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
	// if (inode* parent = single_parent(a, label))
	// {
	// 	return parent;
	// }
	return coord_func(args, smap, shaper, TRANSPOSE);
}

varptr flip (const varptr a, std::vector<size_t> dims)
{
	return flip(a, constant::get<uint64_t>(std::vector<uint64_t>(dims.begin(), dims.end()), 
		tensorshape(std::vector<size_t>{dims.size()})));
}

varptr flip (const varptr a, const varptr dims)
{
	if (nullptr == a.get() || nullptr == dims.get()) return nullptr;
	std::string label = nnutils::formatter() << "flip_" << dims;
	// if (inode* parent = single_parent(a, label))
	// {
	// 	return parent;
	// }
	return coord_func({a, dims}, 
	[](tensorshape outshape, const tensorshape, std::vector<uint64_t> dims)
	{
		// assert inshape.is_compatible_with(outshape)
		std::vector<size_t> slist = outshape.as_list();
		// populate index
		size_t n = outshape.n_elems();
		std::vector<size_t> index(n);
		std::vector<size_t> coord;
		for (size_t i = 0; i < n; ++i)
		{
			coord = outshape.coord_from_idx(i);
			for (size_t d : dims)
			{
				coord[d] = slist[d] - coord[d] - 1;
			}
			index[i] = outshape.flat_idx(coord);
		}
		return index;
	},
	[](tensorshape inshape, std::vector<uint64_t>) { return inshape; }, FLIP);
}



varptr arg_max (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, "argmax"))
	{
		return parent;
	}
	return agg_func(a, "argmax", ARGMAX,
	[](std::vector<std::pair<inode*,inode*> >) -> varptr
	{
		throw std::exception();
	});
}

varptr reduce_max (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, "max"))
	{
		return parent;
	}
	return agg_func(a, "max", RMAX,
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
	if (nullptr == a.get()) return nullptr;
	// always check if the same operation on input exists
	if (inode* parent = single_parent(a, "sum"))
	{
		return parent;
	}
	return agg_func(a, "sum", RSUM,
	[](std::vector<std::pair<inode*,inode*> > args) -> varptr
	{
		return args.front().second;
	});
}


varptr n_elems (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	std::string opname = "n_elems";
	if (inode* parent = single_parent(a, opname))
	{
		return parent;
	}
	return shape_func({a},
	[](tensorshape inshape, std::vector<uint64_t>)
	{
		return std::vector<size_t>{inshape.n_elems()};
	},
	[](tensorshape, std::vector<uint64_t>)
	{
		return tensorshape(std::vector<size_t>{1});
	}, N_ELEMS);
}

varptr n_dimension (const varptr a, size_t dimension)
{
	return n_dimension(a, constant::get<uint64_t>(dimension));
}

varptr n_dimension (const varptr a, const varptr dimension)
{
	if (nullptr == a.get() || nullptr == dimension.get()) return nullptr;
	// std::string opname = nnutils::formatter() << "n_dimension_" << dimension;
	// if (inode* parent = single_parent(a, opname))
	// {
	// 	return parent;
	// }
	return shape_func({a, dimension},
	[](tensorshape inshape, std::vector<uint64_t> dimension)
	{
		assert(dimension.size() > 0);
		std::vector<size_t> vec = inshape.as_list();
		assert(vec.size() > dimension[0]);
		return std::vector<size_t>{vec[dimension[0]]};
	},
	[](tensorshape, std::vector<uint64_t>)
	{
		return tensorshape(std::vector<size_t>{1});
	}, N_DIMS);
}



varptr expand (varptr a, size_t n, size_t dim)
{
	return expand(a, varptr(constant::get<uint64_t>(n)), constant::get<uint64_t>(dim));
}

varptr expand (varptr a, varptr n, size_t dim)
{
	return expand(a, n, constant::get<uint64_t>(dim));
}

varptr expand (const varptr a, const varptr n, const varptr dim)
{
	if (nullptr == a.get()) return nullptr;
	std::vector<inode*> deps = {a, n, dim};
	// std::string opname = nnutils::formatter() << "expand_" << dim;
	// if (inode* parent = ordered_parent(deps, opname))
	// {
	// 	return parent;
	// }
	return functor::get(deps,
	[](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		tensor* tens = args[0]->get_tensor();
		assert(nullptr != tens);
		size_t n = expose<double>(args[1])[0]; // todo: make this size_t once shape_func uses size_t
		uint64_t dim = expose<uint64_t>(args[2])[0];
		tensorshape shape = tens->get_shape();
		std::vector<size_t> slist = shape.as_list();
		assert(slist.size() >= dim);
		slist.insert(slist.begin() + dim, n);

		sindex_io* sio = new sindex_io(
		[n](tensorshape outshape, const tensorshape inshape, std::vector<uint64_t> dim)
		{
			assert(dim.size() > 0);
			size_t nelems = outshape.n_elems();
			std::vector<size_t> indices(nelems);
			std::vector<size_t> slist = inshape.as_list();
			auto it = slist.begin();
			size_t outern = inshape.n_elems();
			size_t innern = std::accumulate(it, it + dim[0], 1, std::multiplies<size_t>());
			size_t repeats = outern / innern;
			size_t nexpansion = innern * n;
			auto iit = indices.begin();
			for (size_t j = 0; j < repeats; ++j)
			{
				for (size_t i = 0; i < n; ++i)
				{
					std::iota(iit + i * innern, iit + (i + 1) * innern, j * innern);
				}
				iit = iit + nexpansion;
			}
			return indices;
		});
		src = std::unique_ptr<idata_src>(sio);
		tens->write_to(*sio);
		sio->shape_info(expose<uint64_t>(args[2]));
		
		return new tensor(tensorshape(slist));
	},
	[](inode* wrt, std::vector<inode*> args)
	{
		return args[0]->derive(wrt);
	}, EXPAND);
}

}

#endif
