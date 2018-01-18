//
//  transform.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/operations/operations.hpp"
#include "include/graph/connector/immutable/shape_dep.hpp"
#include "include/tensor/actors/tens_transform.hpp"

#ifdef TENNCOR_TRANSFORM_HPP

namespace nnet
{

varptr l2norm (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		double l2norm = 0;
		std::vector<double> acvec = expose<double>(aconst);
		for (double acv : acvec)
		{
			l2norm += acv * acv;
		}
		return constant::get(l2norm);
	}
	std::string opname = "l2norm";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a},
	[](std::vector<tensorshape>) { return std::vector<size_t>{1}; },
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_l2norm<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_l2norm<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return l2norm(grad);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr transpose (const varptr a, std::pair<size_t,size_t> axis_swap)
{
	if (nullptr == a.get()) return nullptr;
	// order the axis by min, max
	if (axis_swap.first> axis_swap.second)
	{
		std::swap(axis_swap.first, axis_swap.second);
	}
	std::string opname = nnutils::formatter() << "transpose_" << axis_swap.first << "_" << axis_swap.second;
	// avoid double consecutive transposes (along the same axis)
	if (iconnector* aconn = dynamic_cast<iconnector*>(a.get()))
	{
		std::vector<inode*> childargs = aconn->get_arguments();
		if (0 == a->get_label().compare(opname) && 1 == childargs.size())
		{
			return childargs[0];
		}
	}
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	return immutable::get(std::vector<inode*>{a},
	[axis_swap](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape ts = shapes[0];
		if (ts.is_fully_defined())
		{
			std::vector<size_t> inl = ts.as_list();
			if (axis_swap.second>= inl.size())
			{
				inl.insert(inl.end(), axis_swap.second - inl.size() + 1, 1);
			}
			std::swap(inl[axis_swap.first], inl[axis_swap.second]);
			return inl;
		}
		return tensorshape();
	},
	new actor_func(
	CONN_ACTOR([axis_swap](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_transpose<double>(dest, srcs, axis_swap);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_transpose<signed>(dest, srcs, axis_swap);
			default:
			break;
		}
		return nullptr;
	})),
	[axis_swap](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return transpose(grad, axis_swap);
	}, opname);
}

varptr fit (const varptr a, const varptr watch)
{
	if (nullptr == a.get() || nullptr == watch.get()) return nullptr;
	constant* aconst = dynamic_cast<constant*>(a.get());
	if (aconst && *aconst == (double)0) return a;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
	std::string opname = "fit";
	if (inode* parent = ordered_binary_parent_search(a.get(), watch.get(), opname))
	{
		return parent;
	}
	return immutable::get(std::vector<inode*>{a, watch},
	[](std::vector<tensorshape> shapes) -> tensorshape
	{
		return shapes[1]; // watch is always argument 2
	},
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_fit<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_fit<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return grad;
	}, opname, watch);
}

varptr extend (const varptr a, size_t index, size_t multiplier)
{
	if (nullptr == a.get()) return nullptr;
	if (multiplier == 0) return constant::get(0);
	if (multiplier == 1) return a;
	std::string opname = nnutils::formatter() << "extend_" << index << "_" << multiplier;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	return immutable::get(std::vector<inode*>{a},
	[index, multiplier](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape ts = shapes[0];
		ts.assert_is_fully_defined();
		std::vector<size_t> tv = ts.as_list();
		// allocated additional space along index
		size_t dims = ts.rank();
		if (index>= dims)
		{
			// extending extra dimensions
			size_t extra_dims = index - dims;
			if (extra_dims)
			{
				tv.insert(tv.end(), extra_dims, 1);
			}
			tv.push_back(multiplier);
		}
		else
		{
			tv[index] *= multiplier;
		}
		return tv;
	},
	new actor_func(
	CONN_ACTOR([index, multiplier](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_extend<double>(dest, srcs, index, multiplier);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_extend<signed>(dest, srcs, index, multiplier);
			default:
			break;
		}
		return nullptr;
	})),
	[index, multiplier](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return grad;
	}, opname);
}

varptr compress (const varptr a, BI_TRANS<double> collector,
	optional<size_t> index, std::string name)
{
	if (nullptr == a.get()) return nullptr;
	std::string imm_name = (bool) index ? nnutils::formatter() << name << "_" << *index : name;
	if (inode* parent = unary_parent_search(a.get(), imm_name))
	{
		return parent;
	}
	SHAPER shaper;
	actor_func* forward;
	if ((bool) index)
	{
		shaper = [index](std::vector<tensorshape> shapes) -> tensorshape
		{
			size_t idx = *index;
			tensorshape& ts = shapes[0];
			ts.assert_is_fully_defined();

			size_t srank = ts.rank();
			if (idx>= srank)
			{
				return ts;
			}
			std::vector<size_t> tv = ts.as_list();
			if (1 == srank)
			{
				tv[0] = 1;
			}
			else if (0 == idx)
			// pop front
			{
				tv = std::vector<size_t>(tv.begin()+1, tv.end());
			}
			else if (tv.size()-1 == idx)
			{
				tv.pop_back();
			}
			else
			{
				tv[idx] = 1;
			}
			return tensorshape(tv);
		};

		forward = new actor_func(
		CONN_ACTOR([index, collector](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
			tenncor::tensor_proto::tensor_t type) -> itens_actor*
		{
			switch (type)
			{
				case tenncor::tensor_proto::DOUBLE_T:
					return new tens_compress<double>(dest, srcs, *index, collector);
				case tenncor::tensor_proto::SIGNED_T:
					return new tens_compress<signed>(dest, srcs, *index, collector);
				default:
				break;
			}
			return nullptr;
		}));
	}
	else
	{
		shaper = [](std::vector<tensorshape>) -> tensorshape { return std::vector<size_t>{1}; };
		// scalar shape
		forward = new actor_func(
		CONN_ACTOR([collector](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
		{
			switch (type)
			{
				case tenncor::tensor_proto::DOUBLE_T:
					return new tens_compress<double>(dest, srcs, collector);
				case tenncor::tensor_proto::SIGNED_T:
					return new tens_compress<signed>(dest, srcs, collector);
				default:
				break;
			}
			return nullptr;
		}));
	}
	return immutable::get(std::vector<inode*>{a}, shaper, forward,
	[index](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr bnode = args.front().second;
		if (index)
		{
			if (bnode.get() == bnode.get())
			{
				bnode = identity(bnode);
			}
			bnode->set_metadata("grouping", *index);
		}
		return bnode;
	}, imm_name);
}

varptr reduce_max (const varptr a, optional<size_t> dimension)
{
	return compress(a,
	[](const double left, const double right) -> double
	{
		return std::max(left, right);
	}, dimension, "reduce_max");
}

varptr reduce_sum (const varptr a, optional<size_t> dimension)
{
	return compress(a,
	[](const double left, const double right) -> double
	{
		return left + right;
	}, dimension, "reduce_sum");
}

varptr reduce_mean (const varptr a, optional<size_t> dimension)
{
	varptr denom;
	if (dimension)
	{
		denom = shape_dep::get(a,
		[dimension](tensorshape& s) -> std::vector<size_t>
		{
			return { s.as_list()[*dimension] };
		}, std::vector<size_t>{1},
		nnutils::formatter() << "axis_" << *dimension << "_size");
	}
	else
	{
		denom = shape_dep::get(a,
		[](tensorshape& s) -> std::vector<size_t>
		{
			return { s.n_elems() };
		}, std::vector<size_t>{1}, "shape_nelems");
	}
	return reduce_sum(a, dimension) / denom;
}

varptr arg_compress (const varptr a, REDUCE<double> search,
	optional<size_t> dimension, std::string name)
{
	if (nullptr == a.get()) return nullptr;
	std::string imm_name = (bool) dimension ? nnutils::formatter() << name << "_" << *dimension : name;
	if (inode* parent = unary_parent_search(a.get(), imm_name))
	{
		return parent;
	}
	SHAPER shaper;
	actor_func* forward;
	if (dimension)
	{
		shaper = [dimension](std::vector<tensorshape> shapes) -> tensorshape
		{
			size_t dim = *dimension;
			tensorshape ts = shapes[0];
			ts.assert_is_fully_defined();
			if (dim>= ts.rank())
			{
				throw std::logic_error(nnutils::formatter()
				<< "attempting to obtain arg index along dimension "
				<< dim << " on a " << ts.rank() << " tensor");
			}
			std::vector<size_t> tv = ts.as_list();
			tv[dim] = 1;
			if (tv.size()> 1)
			{
				if (0 == dim)
					// pop front
				{
					tv = std::vector<size_t>(tv.begin()+1, tv.end());
				}
				else if (tv.size()-1 == dim)
				{
					tv.pop_back();
				}
			}
			return tv;
		};
		forward = new actor_func(
		CONN_ACTOR([dimension, search](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
		{
			switch (type)
			{
				case tenncor::tensor_proto::DOUBLE_T:
					return new tens_argcompress<double>(dest, srcs, *dimension, search);
				case tenncor::tensor_proto::SIGNED_T:
					return new tens_argcompress<signed>(dest, srcs, *dimension, search);
				default:
				break;
			}
			return nullptr;
		}));
	}
	else
	{
		shaper = [](std::vector<tensorshape> inshapes) -> tensorshape
		{
			return std::vector<size_t>{inshapes[0].rank()};
		};
		// scalar shape
		forward = new actor_func(
		CONN_ACTOR([search](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
		{
			switch (type)
			{
				case tenncor::tensor_proto::DOUBLE_T:
					return new tens_argcompress<double>(dest, srcs, search);
				case tenncor::tensor_proto::SIGNED_T:
					return new tens_argcompress<signed>(dest, srcs, search);
				default:
				break;
			}
			return nullptr;
		}));
	}
	return immutable::get(std::vector<inode*>{a}, shaper, forward,
	[](std::vector<std::pair<inode*,inode*>>)
	{
		// arg_compression's gradient has no intrinsic meaning
		throw std::logic_error("attempting to get gradient of arg compression: undefined and meaningless operation");
		return nullptr;
	}, imm_name);
}

varptr arg_max (const varptr a, optional<size_t> dimension)
{
	return arg_compress(a,
	[](std::vector<double> data) -> double
	{
		auto mit = std::max_element(data.begin(), data.end(), [](double a, double b)->bool { return a < b; });
		return std::distance(data.begin(), mit);
	}, dimension, "arg_max");
}

varptr flip (const varptr a, std::vector<size_t> dims)
{
	if (nullptr == a.get()) return nullptr;
	std::unordered_set<inode*> audience;
	std::stringstream ss;
	ss << "flip";
	for (size_t d : dims)
	{
		ss << "_" << d;
	}
	std::string opname = ss.str();
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}

	immutable* sym = immutable::get(std::vector<inode*>{a},
	[](std::vector<tensorshape> shapes) -> tensorshape
	{
		return shapes[0];
	},
	new actor_func(
	CONN_ACTOR([dims](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_flip<double>(dest, srcs, dims);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_flip<signed>(dest, srcs, dims);
			default:
			break;
		}
		return nullptr;
	})),
	[dims](std::vector<std::pair<inode*,inode*>> args)
	{
		return flip(args.front().second, dims);
	}, opname);
	return sym;
}

varptr cross_corr2d (const varptr a, const varptr filter,
	std::pair<size_t, size_t> dims)
{
	if (nullptr == a.get() || nullptr == filter.get()) return nullptr;
	std::unordered_set<inode*> audience;
	std::string opname = nnutils::formatter() << "cross_conv_" << dims.first << "_" << dims.second;
	if (inode* parent = ordered_binary_parent_search(a.get(), filter.get(), opname))
	{
		return parent;
	}

	immutable* cv = immutable::get(std::vector<inode*>{a, filter},
	[dims](std::vector<tensorshape> shapes) -> tensorshape
	{
		std::vector<size_t> outshape = shapes[0].as_list();
		std::vector<size_t> filtshape = shapes[1].as_list();
		outshape[dims.first] -= filtshape[dims.first] + 1;
		outshape[dims.second] -= filtshape[dims.second] + 1;
		return outshape;
	},
	new actor_func(
	CONN_ACTOR([dims](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_cross_corr2d<double>(dest, srcs, dims);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_cross_corr2d<signed>(dest, srcs, dims);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>>)
	{
		throw std::bad_function_call(); // NOT IMPLEMENTED
		return constant::get_shared_one();
	}, opname);

	return cv;
}

varptr conv2d (const varptr a, const varptr filter,
	std::pair<size_t, size_t> dims)
{
	return cross_corr2d(a, flip(filter, {dims.first, dims.second}), dims);
}

}

#endif
