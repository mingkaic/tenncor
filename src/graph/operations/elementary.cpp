//
//  elementary.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/operations/operations.hpp"
#include "include/graph/connector/immutable/const_immutable.hpp"

#ifdef TENNCOR_ELEMENTARY_HPP

namespace nnet
{

static inline tensorshape elementary_shaper (std::vector<tensorshape> shapes)
{
	tensorshape lastshape;
	for (size_t i = 0, nshapes = shapes.size(); i < nshapes; ++i)
	{
		if (shapes[i].n_elems() == 1)
		{
			continue;
		}
		if (false == shapes[i].is_compatible_with(lastshape))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shapes[i], ss);
			ss << " is incompatible with shape ";
			print_shape(lastshape, ss);
			throw std::runtime_error(ss.str());
		}
		lastshape = shapes[i];
	}
	if (false == lastshape.is_fully_defined()) return std::vector<size_t>{1};
	return lastshape;
}

static inline SHAPER binary_axial_shape (size_t axis, bool left)
{
	return
	[axis, left](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape shape1; // axial shape
		tensorshape shape2; // real shape
		if (left)
		{
			shape1 = shapes[0];
			shape2 = shapes[1];
		}
		else
		{
			shape1 = shapes[1];
			shape2 = shapes[0];
		}

		if (shape1.n_elems() == 1) return shape2;
		if (shape2.n_elems() == 1) return shape1;

		std::vector<size_t> s2list = shape2.as_list();

		if (axis == 0)
		{
			s2list = std::vector<size_t>(s2list.begin()+1, s2list.end());
		}
		else if (axis == s2list.size() - 1)
		{
			s2list.pop_back();
		}
		else
		{
			s2list[axis] = 1;
		}
		if (false == shape1.is_compatible_with(tensorshape(s2list)))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shape1, ss);
			ss << " is incompatible with shape ";
			print_shape(shape2, ss);
			ss << " along axis " << axis;
			throw std::runtime_error(ss.str());
		}

		if (false == shape2.is_fully_defined()) return std::vector<size_t>{1};
		return shape2;
	};
}

static TRANSFER_FUNC<double> binary_elem (AGGREGATE aggregate)
{
	return [aggregate](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		size_t n_out = shapes.outs_.n_elems();
		size_t n_left = shapes.ins_[0].n_elems();
		size_t n_right = shapes.ins_[1].n_elems();
		bool left_mul = n_left> 1;
		bool right_mul = n_right> 1;
		for (size_t i = 0; i < n_out; i++)
		{
			dest[i] = aggregate(srcs[0][i * left_mul], srcs[1][i * right_mul]);
		}
	};
}

static TRANSFER_FUNC<double> binary_axial (AGGREGATE aggregate, size_t axis, bool left)
{
	short idx = 0;
	if (!left)
	{
		aggregate = [aggregate](double left_arg, double right_arg)
		{
			return aggregate(right_arg, left_arg);
		};
		idx = 1;
	}
	return [aggregate, axis, idx](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::vector<size_t> olist = shapes.outs_.as_list();
		std::vector<size_t> ilist = shapes.ins_[idx].as_list();

		if (1 == shapes.ins_[idx].n_elems())
		{
			for (size_t i = 0; i < n_elems; i++)
			{
				dest[i] = aggregate(srcs[idx][0], srcs[(idx + 1) % 2][i]);
			}
		}
		else
		{
			for (size_t i = 0; i < n_elems; i++)
			{
				size_t axis_idx = i;
				if (axis == 0 && ilist[axis] != olist[axis])
				{
					std::vector<size_t> coords = shapes.outs_.coordinate_from_idx(i);
					if (ilist[axis] != olist[axis+1])
					{
						std::stringstream ss;
						ss << "failed to map ";
						print_shape(shapes.outs_, ss);
						ss << " to ";
						print_shape(shapes.ins_[idx], ss);
						ss << " along axis 0";
						throw std::logic_error(ss.str());
					}
					coords = std::vector<size_t>(coords.begin()+1, coords.end());
					axis_idx = shapes.ins_[idx].flat_idx(coords);
				}
				else if (axis>= ilist.size() || (ilist[axis] == 1 && olist[axis]> 1))
				{
					std::vector<size_t> coords = shapes.outs_.coordinate_from_idx(i);
					coords[axis] = 0;
					axis_idx = shapes.ins_[idx].flat_idx(coords);
				}
				dest[i] = aggregate(srcs[idx][axis_idx], srcs[(idx + 1) % 2][i]);
			}
		}
	};
}

static void axial_set_jacobian (varptr& root, const varptr& branch, size_t axis)
{
	if (iconnector* iconn = dynamic_cast<iconnector*>(root.get()))
	{
		std::unordered_set<ileaf*> temp = branch->get_leaves();
		std::vector<variable*> leef;
		for (ileaf* ilef : temp)
		{
			if (variable* var = dynamic_cast<variable*>(ilef))
			{
				leef.push_back(var);
			}
		}
		iconn->set_jacobian_front(
		[axis](inode* root, std::vector<inode*>, std::vector<inode*>) -> inode*
		{
			return reduce_sum(varptr(root), axis);
		}, leef);
	}
}

static varptr add_helper (const varptr& a, const varptr& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<double> Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (aconst && *aconst == 0)
	{
		return b;
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(aconst);
		return outconst[0] + b;
	}
	else if (bconst && *bconst == 0)
	{
		return a;
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(bconst);
		return a + outconst[0];
	}

	if (inode* parent = unordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a, b}, shaper, new transfer_func<double>(Nf), ginit, opname);
	if (left_axis)
	{
		size_t axis = left_axis->second;
		if (left_axis->first)
		{
			axial_set_jacobian(out, a, axis);
		}
		else
		{
			axial_set_jacobian(out, b, axis);
		}
	}
	return out;
}

static varptr sub_helper (const varptr& a, const varptr& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<double> Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (a.get() == b.get())
	{
		return constant::get(0);
	}
	else if (aconst && *aconst == 0)
	{
		return -b;
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(aconst);
		return outconst[0] - b;
	}
	else if (bconst && *bconst == 0)
	{
		return a;
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(bconst);
		return a - outconst[0];
	}

	if (inode* parent = ordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a, b}, shaper, new transfer_func<double>(Nf), ginit, opname);
	if (left_axis)
	{
		size_t axis = left_axis->second;
		if (left_axis->first)
		{
			axial_set_jacobian(out, a, axis);
		}
		else
		{
			axial_set_jacobian(out, b, axis);
		}
	}
	return out;
}

static varptr mul_helper (const varptr& a, const varptr& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<double> Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (a.get() == b.get())
	{
		return pow(a, 2);
	}
	else if (aconst && *aconst == 0)
	{
		return constant::get(0);
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(aconst);
		return outconst[0] * b;
	}
	else if (bconst && *bconst == 0)
	{
		return constant::get(0);
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(b);
		return a * outconst[0];
	}

	if (inode* parent = unordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a, b}, shaper, new transfer_func<double>(Nf), ginit, opname);
	if (left_axis)
	{
		size_t axis = left_axis->second;
		if (left_axis->first)
		{
			axial_set_jacobian(out, a, axis);
		}
		else
		{
			axial_set_jacobian(out, b, axis);
		}
	}
	return out;
}

static varptr div_helper (const varptr& a, const varptr& b,
	std::string opname, SHAPER shaper, TRANSFER_FUNC<double> Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (a.get() == b.get())
	{
		return constant::get(1);
	}
	else if (aconst && *aconst == 0)
	{
		return constant::get(0);
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(aconst);
		return outconst[0] / b;
	}
	else if (bconst && *bconst == 0)
	// don't allow infinity
	{
		throw std::logic_error("divide by constant node of value zero");
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(bconst);
		return a / outconst[0];
	}

	if (inode* parent = ordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a, b}, shaper, new transfer_func<double>(Nf), ginit, opname);
	if (left_axis)
	{
		size_t axis = left_axis->second;
		if (left_axis->first)
		{
			axial_set_jacobian(out, a, axis);
		}
		else
		{
			axial_set_jacobian(out, b, axis);
		}
	}
	return out;
}


// samples cannot be backpropogated, so we need a special handler
static varptr sample_back_prop (std::vector<std::pair<inode*,inode*>>)
{
//	throw std::bad_function_call();
	// todo: problem: we're evaluating the gradient of something that might not be used
	// solution 1: force a lazy evaluation of grad nodes (hard)
	// solution 2: return a node that errors upon usage (not memory efficient)
	return nullptr;
}

varptr identity (varptr x)
{
	if (nullptr == x.get()) return nullptr;
	std::string opname = "identity";
	if (inode* parent = unary_parent_search(x.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{x}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::memcpy(dest, src[0], sizeof(double) * n_elems);
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return grad;
	}, opname);
	out->extract_metadata(x.get());
	return out;
}

varptr as_constant (varptr x)
{
	if (nullptr == x.get()) return nullptr;
	std::string opname = "constant_immutable";
	if (inode* parent = unary_parent_search(x.get(), opname))
	{
		return parent;
	}
	varptr out = const_immutable::get(x);
	out->extract_metadata(x.get());
	return out;
}

varptr operator + (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::abs(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "abs";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return +data; });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return +grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator - (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = -acv;
		}
		return constant::get(acvec, aconst->get_shape());
	}
	else if (iconnector* aconn = dynamic_cast<iconnector*>(a.get()))
	{
		// avoids double negatives
		std::vector<inode*> childargs = aconn->get_arguments();
		if (0 == a->get_label().compare("neg") && 1 == childargs.size())
		{
			return childargs[0];
		}
	}
	std::string opname = "neg";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return -data; });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return -grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr sin (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::sin(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "sin";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::sin(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sin'(f(x)) = f'(x)*cos(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * cos(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr cos (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::cos(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "cos";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::cos(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// cos'(f(x)) = -f'(x)*sin(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return -grad * sin(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr tan (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::tan(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "tan";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::tan(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sec'(f(x)) = f'(x)*sec^2(f(x))
		// better with = f'(x)/cos^2(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		varptr denom = cos(a);
		return grad / (denom * denom);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr csc (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = 1 / std::sin(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "csc";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return 1 / std::sin(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// csc'(f(x)) = -f'(x)*csc(f(x))*cot(f(x))
		// better with -f'(x)/(sin(f(x)*tan(f(x))))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return -grad / (sin(a) * tan(a));
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr sec (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = 1/std::cos(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "sec";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return 1 / std::cos(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sec'(f(x)) = f'(x)*tan(f(x))*sec(f(x))
		// better with f'(x)*tan(f(x))/cos(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * tan(a) / cos(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr cot (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::cos(acv) / std::sin(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "cot";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::cos(data) / std::sin(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// cot'(f(x)) = -f'(x)*csc^2(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		varptr b = csc(a);
		return -grad * b * b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr exp (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::exp(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "exp";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::exp(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// exp'(f(x)) = f'(x)*exp(f(x))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * exp(a);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr sqrt (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::sqrt(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "sqrt";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::sqrt(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// sqrt'(f(x)) = f'(x)/(2*sqrt(f(x)))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / (2 * sqrt(a));
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr round (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::round(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "round";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::round(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// round'(f(x)) = round(f'(x))
		varptr grad = args.front().second;
		return nnet::round(grad);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr log (const varptr a)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::log(acv);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = "log";
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[](const double data) { return std::log(data); });
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// log'(f(x)) = f'(x) / f(x)
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad / a;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr pow (const varptr a, double scalar)
{
	if (nullptr == a.get()) return nullptr;
	if (scalar == 0)
	{
		return constant::get(1);
	}
	else if (scalar == 1)
	{
		return a;
	}
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = std::pow(acv, scalar);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "pow_" << scalar;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[scalar](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[scalar](const double data) { return std::pow(data, scalar); });
	}),
	[scalar](std::vector<std::pair<inode*,inode*>> args)
	{
		// sqrt'(f(x)) = f'(x) * (scalar*f(x)^(scalar-1))
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return scalar * grad * pow(a, scalar-1);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr clip_val (const varptr a, double min, double max)
{
	if (nullptr == a.get()) return nullptr;
	if (min> max)
	{
		std::swap(min, max);
	}
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			if (min> acv) acv = min;
			else if (max < acv) acv = max;
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "clip_val_" << min << "_" << max;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[min, max](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		std::transform(src[0], src[0] + n_elems, dest,
		[min, max](const double data)
		{
			if (min> data) return min;
			else if (max < data) return max;
			return data;
		});
	}),
	[min, max](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * clip_val(a, min, max);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

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
	new transfer_func<double>(
	[](double* dest, std::vector<const double*> src, shape_io shape)
	{
		assert(1 == src.size());
		size_t n_elems = shape.outs_.n_elems();
		// l2norm = sqrt(sum_i=0:n(sqr(xi)))
		dest[0] = std::sqrt(std::accumulate(src[0], src[0] + n_elems, 0,
		[](const double left, const double right)
		{
			return left + right * right;
		}));
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr grad = args.front().second;
		return l2norm(grad);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr clip_norm (const varptr a, double cap)
{
	assert(cap> 0); // todo: maybe throw to indicate usage error
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		double l2norm = 0;
		std::vector<double> acvec = expose<double>(aconst);
		for (double acv : acvec)
		{
			l2norm += acv * acv;
		}
		for (double& acv : acvec)
		{
			if (l2norm> cap)
			{
				// normalize
				acv = acv * cap / l2norm;
			}
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "clip_l2norm_" << cap;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{l2norm(a), a},
	elementary_shaper,
	new transfer_func<double>(
	[cap](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		double l2norm = srcs[0][0];
		size_t n_out = shapes.outs_.n_elems();
		std::transform(srcs[1], srcs[1] + n_out, dest,
		[cap, l2norm](const double data)
		{
			return data * cap / l2norm;
		});
	}),
	[cap](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr a = args.front().first;
		varptr grad = args.front().second;
	   	return grad * clip_norm(a, cap);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr conditional (double a, const varptr b, std::function<bool(double,double)> compare, std::string name)
{
	if (nullptr == b.get()) return nullptr;
	if (constant* bconst = dynamic_cast<constant*>(b.get()))
	{
		std::vector<double> bcvec = expose<double>(bconst);
		for (double& bcv : bcvec)
		{
			bcv = compare(a, bcv);
		}
		return constant::get(bcvec, bconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "conditional_" << name << "_" << a;
	if (inode* parent = unary_parent_search(b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{b}, elementary_shaper,
	new transfer_func<double>(
	[compare, a](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_out, dest,
		[compare, a](const double data)
		{
			return compare(a, data);
		});
	}),
	[compare, name](std::vector<std::pair<inode*,inode*>> args)
	{
		// todo: consider correctness
		varptr gradb = args[0].second;
		return conditional(0, gradb, compare, name);
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

varptr conditional (const varptr a, double b, std::function<bool(double,double)> compare, std::string name)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = compare(acv, b);
		}
		return constant::get(acvec, aconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "conditional_" << name << "_" << b;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[compare, b](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_out, dest,
		[compare, b](const double data)
		{
			return compare(data, b);
		});
	}),
	[compare, name](std::vector<std::pair<inode*,inode*>> args)
	{
		// todo: consider correctness
		varptr grada = args[0].second;
		return conditional(grada, 0, compare, name);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr conditional (const varptr a, const varptr b, std::function<bool(double,double)> compare, std::string name)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(aconst);
		return conditional(outconst[0], b, compare, name);
	}
	else if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(bconst);
		return conditional(a, outconst[0], compare, name);
	}
	std::string opname = "conditional_" + name;
	if (inode* parent = ordered_binary_parent_search(a.get(), b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a, b}, elementary_shaper,
	new transfer_func<double>(binary_elem(
	AGGREGATE([compare](double left, double right) -> double
	{
		return compare(left, right);
	}))),
	[compare, name](std::vector<std::pair<inode*,inode*>> args)
	{
		assert(args.size() == 2);
		varptr grada = args[0].second;
		varptr gradb = args[1].second;
		// todo: consider correctness
		return conditional(grada, gradb, compare, name);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr eq (const varptr a, const varptr b)
{
	return conditional(a, b, [](double left, double right) { return left == right; }, "eq");
}

varptr neq (const varptr a, const varptr b)
{
	return conditional(a, b, [](double left, double right) { return left != right; }, "neq");
}

varptr binomial_sample (double n, const varptr p)
{
	if (nullptr == p.get()) return nullptr;
	if (constant* pconst = dynamic_cast<constant*>(p.get()))
	{
		std::default_random_engine& gen = nnutils::get_generator();
		std::vector<double> pcvec = expose<double>(pconst);
		std::vector<double> pvec(0, pcvec.size());
		std::transform(pcvec.begin(), pcvec.end(), pvec.begin(),
		[&gen, n](double pcv)
		{
			std::binomial_distribution<int> dist(n, pcv);
			return dist(gen);
		});
		return constant::get(pvec, pconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "binomial_sample_n" << n;
	if (inode* parent = unary_parent_search(p.get(), opname))
	{
		return parent;
	}
	return immutable::get(std::vector<inode*>{p}, elementary_shaper,
	new transfer_func<double>([n](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::default_random_engine& gen = nnutils::get_generator();
		std::transform(srcs[0], srcs[0] + n_out, dest,
		[&gen, n](const double p)
		{
			assert(p>= 0 && p <= 1);
			std::binomial_distribution<int> dist(n, p);
			return dist(gen);
		});
	}), sample_back_prop, opname);
}

varptr binomial_sample (const varptr n, double p)
{
	if (nullptr == n.get()) return nullptr;
	assert(p>= 0 && p <= 1);
	if (constant* nconst = dynamic_cast<constant*>(n.get()))
	{
		std::default_random_engine& gen = nnutils::get_generator();
		std::vector<double> ncvec = expose<double>(nconst);
		std::vector<double> nvec(0, ncvec.size());
		std::transform(ncvec.begin(), ncvec.end(), nvec.begin(),
		[&gen, p](double ncv)
		{
			std::binomial_distribution<int> dist(ncv, p);
			return dist(gen);
		});
		return constant::get(nvec, nconst->get_shape());
	}
	std::string opname = nnutils::formatter() << "binomial_sample_p" << p;
	if (inode* parent = unary_parent_search(n.get(), opname))
	{
		return parent;
	}
	return immutable::get(std::vector<inode*>{n}, elementary_shaper,
	new transfer_func<double>([p](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::default_random_engine& gen = nnutils::get_generator();
		std::transform(srcs[0], srcs[0] + n_out, dest,
		[&gen, p](const double n)
		{
		   std::binomial_distribution<int> dist(n, p);
		   return dist(gen);
		});
	}), sample_back_prop, opname);
}

varptr binomial_sample (const varptr n, const varptr p)
{
	if (nullptr == n.get() || nullptr == p.get()) return nullptr;
	constant* nconst = dynamic_cast<constant*>(n.get());
	constant* pconst = dynamic_cast<constant*>(p.get());
	if (nconst && 1 == nconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(nconst);
		return binomial_sample(outconst[0], p);
	}
	else if (pconst && 1 == pconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(pconst);
		return binomial_sample(n, outconst[0]);
	}
	std::string opname = nnutils::formatter() << "binomial_sample";
	if (inode* parent = ordered_binary_parent_search(n.get(), p.get(), opname))
	{
		return parent;
	}
	return immutable::get(std::vector<inode*>{p}, elementary_shaper,
	new transfer_func<double>([](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		size_t n_out = shapes.outs_.n_elems();
		std::default_random_engine& gen = nnutils::get_generator();
		for (size_t i = 0; i < n_out; i++)
		{
			size_t n = srcs[0][i];
			double p = srcs[1][i];
			assert(p>= 0 && p <= 1);
			std::binomial_distribution<int> dist(n, p);
			dest[i] = dist(gen);
		}
	}), sample_back_prop, opname);
}

varptr operator + (double a, const varptr b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (a == 0) return b;
	if (constant* bconst = dynamic_cast<constant*>(b.get()))
	{
		if (*bconst == 0)
		{
			return constant::get(a);
		}
		std::vector<double> bcvec = expose<double>(bconst);
		for (double& bcv : bcvec)
		{
			bcv = a + bcv;
		}
		return constant::get(bcvec, bconst->get_shape());
	}
	std::string opname = nnutils::formatter() << a << "_add";
	if (inode* parent = unary_parent_search(b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{b}, elementary_shaper,
	new transfer_func<double>(
	[a](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[a](const double data)
		{
			return a + data;
		});
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(c, g(x)) = g'(x)
		varptr grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

varptr operator + (const varptr a, double b)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		if (*aconst == 0)
		{
			return constant::get(b);
		}
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = acv + b;
		}
		return constant::get(acvec, aconst->get_shape());
	}
	if (b == 0) return a;
	std::string opname = nnutils::formatter() << "add_" << b;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[b](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[b](const double data)
		{
			return data + b;
		});
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = f'(x)
		varptr grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator + (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return add_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return add_axial_a(a, b, *baxis);
	}
	return add_helper(a, b, "add", elementary_shaper,
	binary_elem(
	AGGREGATE([](const double left, const double right) -> double
	{
		return left + right;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x) + g'(x)
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return ag + bg;
	});
}

varptr operator - (double a, const varptr b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (a == 0) return -b;
	if (constant* bconst = dynamic_cast<constant*>(b.get()))
	{
		if (*bconst == 0)
		{
			return constant::get(a);
		}
		std::vector<double> bcvec = expose<double>(bconst);
		for (double& bcv : bcvec)
		{
			bcv = a - bcv;
		}
		return constant::get(bcvec, bconst->get_shape());
	}
	std::string opname = nnutils::formatter() << a << "_sub";
	if (inode* parent = unary_parent_search(b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{b}, elementary_shaper,
	new transfer_func<double>(
	[a](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[a](const double data)
		{
			return a - data;
		});
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(c, g(x)) = -g'(x)
		varptr grad = args.at(0).second;
		return -grad;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

varptr operator - (const varptr a, double b)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	{
		if (*aconst == 0)
		{
			return constant::get(-b);
		}
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = acv - b;
		}
		return constant::get(acvec, aconst->get_shape());
	}
	if (b == 0) return a;
	std::string opname = nnutils::formatter() << "sub_" << b;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[b](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[b](const double data)
		{
			return data - b;
		});
	}),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = f'(x)
		varptr grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator - (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return sub_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return sub_axial_a(a, b, *baxis);
	}
	return sub_helper(a, b, "sub", elementary_shaper,
	binary_elem(
	AGGREGATE([](const double left, const double right) -> double
	{
		return left - right;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x) - g'(x)
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return ag - bg;
	});
}

varptr operator * (double a, const varptr b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	if (constant* bconst = dynamic_cast<constant*>(b.get()))
	// optimize only applies to constants
	{
		if (*bconst == 0 || 0 == a)
		{
			return constant::get(0);
		}
		if (*bconst == 1)
		{
			return constant::get(a);
		}
		std::vector<double> bcvec = expose<double>(bconst);
		for (double& bcv : bcvec)
		{
			bcv = a * bcv;
		}
		return constant::get(bcvec, bconst->get_shape());
	}
	if (0 == a) return constant::get(0);
	if (1 == a) return b;
	if (-1 == a) return -b;
	std::string opname = nnutils::formatter() << a << "_mul";
	if (inode* parent = unary_parent_search(b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{b},
	elementary_shaper,
	new transfer_func<double>(
	[a](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[a](const double data)
		{
			return a * data;
		});
	}),
	[a](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(c, g(x)) = c*g'(x)
		varptr grad = args.at(0).second;
		return a * grad;
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

varptr operator * (const varptr a, double b)
{
	if (nullptr == a.get()) return nullptr;
	if (constant* aconst = dynamic_cast<constant*>(a.get()))
	// optimize only applies to constants
	{
		if (*aconst == 0 || 0 == b)
		{
			return constant::get(0);
		}
		if (*aconst == 1)
		{
			return constant::get(b);
		}
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = acv * b;
		}
		return constant::get(acvec, aconst->get_shape());
	}
	if (0 == b) return constant::get(0);
	if (1 == b) return a;
	if (-1 == b) return -a;
	std::string opname = nnutils::formatter() << "mul_" << b;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a},
	elementary_shaper,
	new transfer_func<double>(
	[b](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[b](const double data)
		{
			return data * b;
		});
	}),
	[b](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = c*f'(x)
		varptr grad = args.at(0).second;
		return b * grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator * (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return mul_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return mul_axial_a(a, b, *baxis);
	}
	return mul_helper(a, b, "mul", elementary_shaper,
	binary_elem(
	AGGREGATE([](const double left, const double right) -> double
	{
		return left * right;
	})),
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

varptr operator / (double a, const varptr b)
{
	if (nullptr == b.get()) return nullptr;
	// we don't want to return constant a otherwise it could leak if we're returning root
	// (roots will never have an audience, so it will never self-destroy)
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (bconst)
	// optimize only applies to constants
	{
		if (*bconst == 0)
		{
			throw std::logic_error("divide by constant node of value zero");
		}
		if (*bconst == 1)
		{
			return constant::get(a);
		}
		std::vector<double> bcvec = expose<double>(bconst);
		for (double& bcv : bcvec)
		{
			bcv = a / bcv;
		}
		return constant::get(bcvec, bconst->get_shape());
	}
	if (a == 0)
	{
		return constant::get(0);
	}
	std::string opname = nnutils::formatter() << a << "_div";
	if (inode* parent = unary_parent_search(b.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{b}, elementary_shaper,
	new transfer_func<double>(
	[a](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[a](const double data)
		{
			return a / data;
		});
	}),
	[a](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(c, g(x)) = -c*g'(x)/g^2(x)
		varptr b = args.at(0).first;
		varptr bg = args.at(0).second;
		return -a * bg / (b * b);
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

varptr operator / (const varptr a, double b)
{
	if (nullptr == a.get()) return nullptr;
	constant* aconst = dynamic_cast<constant*>(a.get());
	if (aconst)
	{
		if (*aconst == 0)
		{
			return constant::get(0);
		}
		std::vector<double> acvec = expose<double>(aconst);
		for (double& acv : acvec)
		{
			acv = acv / b;
		}
		return constant::get(acvec, aconst->get_shape());
	}
	if (b == 0)
	{
		throw std::logic_error("divide by zero");
	}
	if (b == 1) return a;
	std::string opname = nnutils::formatter() << "div_" << b;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, elementary_shaper,
	new transfer_func<double>(
	[b](double* dest, std::vector<const double*> srcs, shape_io shapes)
	{
		assert(1 == srcs.size());
		size_t n_elems = shapes.outs_.n_elems();
		std::transform(srcs[0], srcs[0] + n_elems, dest,
		[b](const double data)
		{
			return data / b;
		});
	}),
	[b](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = f'(x)/c
		varptr ag = args.at(0).second;
		return ag / b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator / (const varptr a, const varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	optional<size_t> aaxis = a->get_metadata("grouping");
	optional<size_t> baxis = b->get_metadata("grouping");
	if (aaxis)
	{
		if (false == (bool)baxis || *aaxis == *baxis)
		{
			return div_axial_b(a, b, *aaxis);
		}
	}
	else if (baxis)
	{
		return div_axial_a(a, b, *baxis);
	}
	return div_helper(a, b, "div", elementary_shaper,
	binary_elem(
	AGGREGATE([](const double left, const double right) -> double
	{
		return left / right;
	})),
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

varptr add_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return add_helper(a, b, nnutils::formatter() << "add_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left + right;
	}), axis_a, true),
	[axis_a](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return add_axial_a(ag, bg, axis_a);
	}, std::pair<bool, size_t>{ true, axis_a });
}

varptr add_axial_b (const varptr a, const varptr b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return add_helper(a, b, nnutils::formatter() << "add_axis_b_" << axis_b,
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left + right;
	}), axis_b, false),
	[axis_b](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return add_axial_b(ag, bg, axis_b);
	}, std::pair<bool, size_t>{ false, axis_b });
}

varptr sub_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return sub_helper(a, b, nnutils::formatter() << "sub_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left - right;
	}), axis_a, true),
	[axis_a](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return sub_axial_a(ag, bg, axis_a);
	}, std::pair<bool, size_t>{ true, axis_a });
}

varptr sub_axial_b (const varptr a, const varptr b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return sub_helper(a, b, nnutils::formatter() << "sub_axis_b_" << axis_b,
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left - right;
	}), axis_b, false),
	[axis_b](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return sub_axial_b(ag, bg, axis_b);
	}, std::pair<bool, size_t>{ false, axis_b });
}

varptr mul_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return mul_helper(a, b, nnutils::formatter() << "mul_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left * right;
	}), axis_a, true),
	[axis_a](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return mul_axial_a(ag, b, axis_a) + mul_axial_a(a, bg, axis_a);
	}, std::pair<bool, size_t>{ true, axis_a });
}

varptr mul_axial_b (const varptr a, const varptr b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return mul_helper(a, b, nnutils::formatter() << "mul_axis_b_" << axis_b,
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left * right;
	}), axis_b, false),
	[axis_b](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = f'(x)*g(x) + f(x)*g'(x)
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return mul_axial_b(ag, b, axis_b) + mul_axial_b(a, bg, axis_b);
	}, std::pair<bool, size_t>{ false, axis_b });
}

varptr div_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return div_helper(a, b, nnutils::formatter() << "div_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left / right;
	}), axis_a, true),
	[axis_a](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return (mul_axial_a(ag, b, axis_a) - mul_axial_b(bg, a, axis_a)) / pow(b, 2);
	}, std::pair<bool, size_t>{ true, axis_a });
}

varptr div_axial_b (const varptr a, const varptr b, size_t axis_b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return div_helper(a, b, nnutils::formatter() << "div_axis_b_" << axis_b,
	binary_axial_shape(axis_b, false),
	binary_axial(AGGREGATE([](double left, double right) -> double
	{
		return left / right;
	}), axis_b, false),
	[axis_b](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), g(x)) = (f'(x)*g(x) - f(x)*g'(x))/g^2(x)
		varptr a = args.at(0).first;
		varptr b = args.at(1).first;
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return (mul_axial_b(ag, b, axis_b) - mul_axial_a(bg, a, axis_b)) / pow(b, 2);
	}, std::pair<bool, size_t>{ false, axis_b });
}

}

#endif
