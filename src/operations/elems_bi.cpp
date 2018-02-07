//
//  elems_bi.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-17.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operations/operations.hpp"

#ifdef TENNCOR_ELEM_BI_HPP

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
	std::string opname, SHAPER shaper, CONN_ACTOR Nf, BACK_MAP ginit,
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
	varptr out = linear::get(std::vector<inode*>{a, b}, shaper, new actor_func(Nf), ginit, opname);
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
	std::string opname, SHAPER shaper, CONN_ACTOR Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (a.get() == b.get())
	{
		switch (a->get_type())
		{
			case DOUBLE:
				return constant::get((double) 0);
			case INT:
				return constant::get((signed) 0);
			default:
				throw std::exception(); // unsupported type
		}
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
	varptr out = linear::get(std::vector<inode*>{a, b}, shaper, new actor_func(Nf), ginit, opname);
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
	std::string opname, SHAPER shaper, CONN_ACTOR Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (a.get() == b.get())
	{
		return pow(a, 2);
	}
	else if (aconst && *aconst == (double) 0)
	{
		switch (a->get_type())
		{
			case DOUBLE:
				return constant::get((double) 0);
			case INT:
				return constant::get((signed) 0);
			default:
				throw std::exception(); // unsupported type
		}
	}
	else if (aconst && 1 == aconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(aconst);
		return outconst[0] * b;
	}
	else if (bconst && *bconst == (double) 0)
	{
		switch (b->get_type())
		{
			case DOUBLE:
				return constant::get((double) 0);
			case INT:
				return constant::get((signed) 0);
			default:
				throw std::exception(); // unsupported type
		}
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
	varptr out = linear::get(std::vector<inode*>{a, b}, shaper, new actor_func(Nf), ginit, opname);
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
	std::string opname, SHAPER shaper, CONN_ACTOR Nf, BACK_MAP ginit,
	optional<std::pair<bool, size_t> > left_axis = optional<std::pair<bool, size_t> >())
{
	constant* aconst = dynamic_cast<constant*>(a.get());
	constant* bconst = dynamic_cast<constant*>(b.get());
	if (a.get() == b.get())
	{
		switch (a->get_type())
		{
			case DOUBLE:
				return constant::get((double) 1);
			case INT:
				return constant::get((signed) 1);
			default:
				throw std::exception(); // unsupported type
		}
	}
	else if (aconst && *aconst == 0)
	{
		switch (a->get_type())
		{
			case DOUBLE:
				return constant::get((double) 0);
			case INT:
				return constant::get((signed) 0);
			default:
				throw std::exception(); // unsupported type
		}
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
	varptr out = linear::get(std::vector<inode*>{a, b}, shaper, new actor_func(Nf), ginit, opname);
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
	return linear::get(std::vector<inode*>{n, p}, elementary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_bin_sample<double>(dest, srcs);
			case INT:
				throw std::exception(); // p can't be signed 
				// todo: modify to allow multi-typal operations
			default:
			break;
		}
		return nullptr;
	})), [](std::vector<std::pair<inode*,inode*> >){ return nullptr; }, opname);
}

varptr pow (const varptr base, const varptr xponent)
{
	if (nullptr == base.get() || nullptr == xponent.get()) return nullptr;
	constant* bconst = dynamic_cast<constant*>(base.get());
	constant* xconst = dynamic_cast<constant*>(xponent.get());
	if (bconst && 1 == bconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(bconst);
		return pow(outconst[0], xponent);
	}
	else if (xconst && 1 == xconst->get_shape().n_elems())
	{
		std::vector<double> outconst = expose<double>(xconst);
		return pow(base, outconst[0]);
	}
	std::string opname = "pow";
	if (inode* parent = ordered_binary_parent_search(base.get(), xponent.get(), opname))
	{
		return parent;
	}
	varptr out = linear::get(std::vector<inode*>{base, xponent}, elementary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_pow<double>(dest, srcs);
			case INT:
				return new tens_pow<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// pow'(f(x), g(x)) = f'(x) * g(x) * pow(f(x), g(x) - 1) +
		//						g'(x) * pow(f(x), g(x)) * log((f(x))
		varptr b = args.at(0).first;
		varptr bg = args.at(0).second;
		varptr x = args.at(1).first;
		varptr xg = args.at(1).second;
		return bg * x * pow(b, x - 1) + xg * pow(b, x) * log(b);
	}, opname);
	out->extract_metadata(xponent.get());
	return out;
}

varptr conditional (const varptr a, const varptr b, COMPARE<double> compare, std::string name)
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
	varptr out = linear::get(std::vector<inode*>{a, b}, elementary_shaper,
	new actor_func(
	CONN_ACTOR([compare](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_conditional<double>(dest, srcs, compare);
			case INT:
				return new tens_conditional<signed>(dest, srcs, compare);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args) -> varptr
	{
		throw std::exception(); // no such gradient
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
	CONN_ACTOR([](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_add<double>(dest, srcs);
			case INT:
				return new tens_add<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	}),
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
	CONN_ACTOR([](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_sub<double>(dest, srcs);
			case INT:
				return new tens_sub<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	}),
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
	CONN_ACTOR([](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_mul<double>(dest, srcs);
			case INT:
				return new tens_mul<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	}),
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
	CONN_ACTOR([](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_div<double>(dest, srcs);
			case INT:
				return new tens_div<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	}),
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

// START DEPRECATE
varptr add_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return add_helper(a, b, nnutils::formatter() << "add_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	CONN_ACTOR([axis_a](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_axial_add<double>(dest, srcs, axis_a);
			case INT:
				return new tens_axial_add<signed>(dest, srcs, axis_a);
			default:
			break;
		}
		return nullptr;
	}),
	[axis_a](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr ag = args.at(0).second;
		varptr bg = args.at(1).second;
		return add_axial_a(ag, bg, axis_a);
	}, std::pair<bool, size_t>{ true, axis_a });
}

varptr add_axial_b (const varptr a, const varptr b, size_t axis_b)
{
	return add_axial_a(b, a, axis_b);
}

varptr sub_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return sub_helper(a, b, nnutils::formatter() << "sub_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	CONN_ACTOR([axis_a](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_axial_sub<double>(dest, srcs, axis_a, true);
			case INT:
				return new tens_axial_sub<signed>(dest, srcs, axis_a, true);
			default:
			break;
		}
		return nullptr;
	}),
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
	CONN_ACTOR([axis_b](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_axial_sub<double>(dest, srcs, axis_b, false);
			case INT:
				return new tens_axial_sub<signed>(dest, srcs, axis_b, false);
			default:
			break;
		}
		return nullptr;
	}),
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
	CONN_ACTOR([axis_a](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_axial_mul<double>(dest, srcs, axis_a);
			case INT:
				return new tens_axial_mul<signed>(dest, srcs, axis_a);
			default:
			break;
		}
		return nullptr;
	}),
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
	return mul_axial_a(b, a, axis_b);
}

varptr div_axial_a (const varptr a, const varptr b, size_t axis_a)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return div_helper(a, b, nnutils::formatter() << "div_axis_a_" << axis_a,
	binary_axial_shape(axis_a, true),
	CONN_ACTOR([axis_a](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_axial_div<double>(dest, srcs, axis_a, true);
			case INT:
				return new tens_axial_div<signed>(dest, srcs, axis_a, true);
			default:
			break;
		}
		return nullptr;
	}),
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
	CONN_ACTOR([axis_b](out_wrapper<void>& dest, 
		std::vector<in_wrapper<void> >& srcs,
		TENS_TYPE type) -> itens_actor*
	{
		switch (type)
		{
			case DOUBLE:
				return new tens_axial_div<double>(dest, srcs, axis_b, false);
			case INT:
				return new tens_axial_div<signed>(dest, srcs, axis_b, false);
			default:
			break;
		}
		return nullptr;
	}),
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
// END DEPRECATE

}

#endif
