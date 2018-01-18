//
//  elem_uni.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/operations/operations.hpp"
#include "include/graph/connector/immutable/const_immutable.hpp"
#include "include/tensor/actors/tens_elem_uni.hpp"

#ifdef TENNCOR_ELEM_UNI_HPP

namespace nnet
{

static inline tensorshape unary_shaper (std::vector<tensorshape> shapes)
{
	return shapes[0];
}

// samples cannot be backpropogated, so we need a special handler
static varptr sample_back_prop (std::vector<std::pair<inode*,inode*> >)
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
	varptr out = immutable::get(std::vector<inode*>{x}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_pipein<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_pipein<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs, 
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_pipein<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_pipein<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_neg<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_neg<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_sin<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_sin<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_cos<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_cos<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_tan<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_tan<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_csc<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_csc<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_sec<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_sec<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_cot<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_cot<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_exp<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_exp<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_sqrt<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_sqrt<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_round<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_round<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_log<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_log<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([scalar](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_pow<double>(dest, srcs, scalar);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_pow<signed>(dest, srcs, scalar);
			default:
			break;
		}
		return nullptr;
	})),
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

varptr clip (const varptr a, double min, double max)
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
	std::string opname = nnutils::formatter() << "clip_" << min << "_" << max;
	if (inode* parent = unary_parent_search(a.get(), opname))
	{
		return parent;
	}
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([min, max](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_clip<double>(dest, srcs, min, max);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_clip<signed>(dest, srcs, min, max);
			default:
			break;
		}
		return nullptr;
	})),
	[min, max](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr a = args.front().first;
		varptr grad = args.front().second;
		return grad * clip(a, min, max);
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
			if (l2norm > cap)
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
	varptr out = immutable::get(std::vector<inode*>{l2norm(a), a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([cap](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		assert(2 == srcs.size());
		const void* l2norm = srcs[0].first;

		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_clip_norm<double>(dest, {srcs[1]}, l2norm, cap);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_clip_norm<signed>(dest, {srcs[1]}, l2norm, cap);
			default:
			break;
		}
		return nullptr;
	})),
	[cap](std::vector<std::pair<inode*,inode*>> args)
	{
		varptr a = args.front().first;
		varptr grad = args.front().second;
	   	return grad * clip_norm(a, cap);
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr conditional (double a, const varptr b, COMPARE<double> compare, std::string name)
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
	UNI_COMP<double> comp = [compare, a](double b)
	{
		return compare(a, b);
	};
	varptr out = immutable::get(std::vector<inode*>{b}, unary_shaper,
	new actor_func(
	CONN_ACTOR([comp](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_conditional_uni<double>(dest, srcs, comp);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_conditional_uni<signed>(dest, srcs, comp);
			default:
			break;
		}
		return nullptr;
	})),
	[compare, name](std::vector<std::pair<inode*,inode*>> args)
	{
		// todo: consider correctness
		varptr gradb = args[0].second;
		return conditional(0, gradb, compare, name);
	}, opname);
	out->extract_metadata(b.get());
	return out;
}

varptr conditional (const varptr a, double b, COMPARE<double> compare, std::string name)
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
	UNI_COMP<double> comp = [compare, b](double a)
	{
		return compare(a, b);
	};
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([comp](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_conditional_uni<double>(dest, srcs, comp);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_conditional_uni<signed>(dest, srcs, comp);
			default:
			break;
		}
		return nullptr;
	})),
	[compare, name](std::vector<std::pair<inode*,inode*>> args)
	{
		// todo: consider correctness
		varptr grada = args[0].second;
		return conditional(grada, 0, compare, name);
	}, opname);
	out->extract_metadata(a.get());
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
	varptr out = immutable::get(std::vector<inode*>{a, b}, unary_shaper,
	new actor_func(
	CONN_ACTOR([compare](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_conditional<double>(dest, srcs, compare);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_conditional<signed>(dest, srcs, compare);
			default:
			break;
		}
		return nullptr;
	})),
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

varptr binomial_sample (signed n, const varptr p)
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
	return immutable::get(std::vector<inode*>{p}, unary_shaper,
	new actor_func(
	CONN_ACTOR([n](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_bin_sample_uni<double>(dest, srcs, n);
			case tenncor::tensor_proto::SIGNED_T:
				throw std::exception(); // p can't be signed
			default:
			break;
		}
		return nullptr;
	})), sample_back_prop, opname);
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
	return immutable::get(std::vector<inode*>{n}, unary_shaper,
	new actor_func(
	CONN_ACTOR([p](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_bin_sample_uni<double>(dest, srcs, p);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_bin_sample_uni<signed>(dest, srcs, p);
			default:
			break;
		}
		return nullptr;
	})), sample_back_prop, opname);
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([b](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_uni_add<double>(dest, srcs, b);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_uni_add<signed>(dest, srcs, b);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = f'(x)
		varptr grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator + (double a, const varptr b)
{
	return b + a;
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([b](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_uni_sub<double>(dest, srcs, b);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_uni_sub<signed>(dest, srcs, b);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = f'(x)
		varptr grad = args.at(0).second;
		return grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
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
	varptr out = immutable::get(std::vector<inode*>{b}, unary_shaper,
	new actor_func(
	CONN_ACTOR([a](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_uni_sub<double>(a, dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_uni_sub<signed>(a, dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(c, g(x)) = -g'(x)
		varptr grad = args.at(0).second;
		return -grad;
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([b](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_uni_mul<double>(dest, srcs, b);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_uni_mul<signed>(dest, srcs, b);
			default:
			break;
		}
		return nullptr;
	})),
	[b](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = c*f'(x)
		varptr grad = args.at(0).second;
		return b * grad;
	}, opname);
	out->extract_metadata(a.get());
	return out;
}

varptr operator * (double a, const varptr b)
{
	return b * a;
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
	varptr out = immutable::get(std::vector<inode*>{a}, unary_shaper,
	new actor_func(
	CONN_ACTOR([b](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_uni_div<double>(dest, srcs, b);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_uni_div<signed>(dest, srcs, b);
			default:
			break;
		}
		return nullptr;
	})),
	[b](std::vector<std::pair<inode*,inode*>> args)
	{
		// h'(f(x), c) = f'(x)/c
		varptr ag = args.at(0).second;
		return ag / b;
	}, opname);
	out->extract_metadata(a.get());
	return out;
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
	varptr out = immutable::get(std::vector<inode*>{b}, unary_shaper,
	new actor_func(
	CONN_ACTOR([a](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_uni_div<double>(a, dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_uni_div<signed>(a, dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
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

}

#endif
