//
//  operator.cpp
//  wire
//

#include "wire/operators.hpp"

#ifdef WIRE_OPERATORS_HPP

namespace wire
{

static inline Identifier* single_parent (
	std::string argid, slip::OPCODE opcode)
{
	FunctorSetT funcs = Graph::get_global().get_func(opcode);
	for (Functor* f : funcs)
	{
		auto args = f->get_args();
		if (args.size() == 1 && args[0] == argid)
		{
			return f;
		}
	}
	return nullptr;
}

static inline Identifier* ordered_parent (
	std::vector<std::string> srcs, slip::OPCODE opcode)
{
	assert(srcs.size() > 0);
	FunctorSetT funcs = Graph::get_global().get_func(opcode);
	for (Functor* f : funcs)
	{
		auto args = f->get_args();
		if (std::equal(args.begin(), args.end(), srcs.begin()))
		{
			return f;
		}
	}
	return nullptr;
}

static inline Identifier* unordered_parent (
	std::unordered_set<std::string> srcs, slip::OPCODE opcode)
{
	assert(srcs.size() > 0);
	FunctorSetT funcs = Graph::get_global().get_func(opcode);
	for (Functor* f : funcs)
	{
		auto args = f->get_args();
		std::unordered_set<std::string> argset(
			args.begin(), args.end());
		if (argset == srcs)
		{
			return f;
		}
	}
	return nullptr;
}

Identifier* cast (Identifier* type, Identifier* a)
{
	if (nullptr == a || nullptr == type)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::CAST;
	if (Identifier* parent = ordered_parent({type->get_uid(), a->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({type, a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return cast(args.front(), args.back()->derive(wrt));
	});
}

Identifier* abs (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ABS;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return abs(args.front()->derive(wrt));
	});
}

Identifier* neg (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::NEG;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return neg(args.front()->derive(wrt));
	});
}

Identifier* logical_not (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::NOT;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return logical_not(args.front()->derive(wrt));
	});
}

Identifier* sin (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::SIN;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// sin'(f) = f'*cos(f)
		auto f = args.front();
		return mul(f->derive(wrt), cos(f));
	});
}

Identifier* cos (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::COS;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// cos'(f) = -f'*sin(f)
		auto f = args.front();
		return mul(neg(f->derive(wrt)), sin(f));
	});
}

Identifier* tan (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::TAN;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// tan'(f) = f'*sec^2(f)
		// 		= f'/cos^2(f)
		auto f = args.front();
		auto denom = cos(f);
		return div(f->derive(wrt), mul(denom, denom));
	});
}

Identifier* exp (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::EXP;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// exp'(f) = f'*exp(f)
		auto f = args.front();
		return mul(f->derive(wrt), exp(f));
	});
}

Identifier* log (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::LOG;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// log'(f) = f' / f
		auto f = args.front();
		return div(f->derive(wrt), f);
	});
}

Identifier* sqrt (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::SQRT;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// sqrt'(f) = f'/(2*sqrt(f))
		auto f = args.front();
		auto denom = sqrt(f);
		return div(f->derive(wrt), add(denom, denom));
	});
}

Identifier* round (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ROUND;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// round'(f) = round(f')
		return round(args.front()->derive(wrt));
	});
}

Identifier* pow (Identifier* b, Identifier* x)
{
	if (nullptr == b || nullptr == x)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::POW;
	if (Identifier* parent = ordered_parent({b->get_uid(), x->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({b, x}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
		//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
		auto f = args.front();
		auto g = args.back();
		assert(g->has_data());
		Identifier* one = make_one(g);
		auto lhs = pow(f, sub(g, one));
		auto rhs = add(mul(f->derive(wrt), g), mul(g->derive(wrt), mul(f, log(f))));
		return mul(lhs, rhs);
	});
}

Identifier* add (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ADD;
	if (Identifier* parent = unordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = f' + g'
		return add(args.front()->derive(wrt), args.back()->derive(wrt));
	});
}

Identifier* sub (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::SUB;
	if (Identifier* parent = ordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = f' - g'
		return sub(args.front()->derive(wrt), args.back()->derive(wrt));
	});
}

Identifier* mul (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::MUL;
	if (Identifier* parent = unordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = f' * g + g' * f
		auto f = args.front();
		auto g = args.back();
		return add(mul(f->derive(wrt), g), mul(g->derive(wrt), f));
	});
}

Identifier* div (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::DIV;
	if (Identifier* parent = ordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = (f' * g - g' * f) / g^2
		//			= f' / g - ((g' * f) / g) / g
		auto f = args.front();
		auto g = args.back();
		auto lhs = div(f->derive(wrt), g);
		auto rhs_num = div(mul(g->derive(wrt), f), g);
		return sub(lhs, div(rhs_num, g));
	});
}

Identifier* eq (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::EQ;
	if (Identifier* parent = unordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return eq(args.front(), args.back());
	});
}

Identifier* neq (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::NE;
	if (Identifier* parent = unordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return neq(args.front(), args.back());
	});
}

Identifier* lt (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::LT;
	if (Identifier* parent = ordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return lt(args.front(), args.back());
	});
}

Identifier* gt (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::GT;
	if (Identifier* parent = ordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return gt(args.front(), args.back());
	});
}

Constant* sample_grad (Identifier*, std::vector<Identifier*> args)
{
	clay::State state = args.front()->get_state();
	unsigned short nbytes = clay::type_size(state.dtype_) *
		state.shape_.n_elems();
	std::shared_ptr<char> data = clay::make_char(nbytes);
	memset(data.get(), 0, nbytes);
	return new Constant(data, state.shape_, state.dtype_, "sample_grad");
}

Identifier* binomial_sample (Identifier* n, Identifier* p)
{
	if (nullptr == n || nullptr == p)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::BINO;
	if (Identifier* parent = ordered_parent({n->get_uid(), p->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({n, p}, opcode, sample_grad);
}

Identifier* binomial_sample (Identifier* n, double p)
{
	return binomial_sample(n, Constant::get(p));
}

Identifier* uniform_sample (Identifier* min, Identifier* max)
{
	if (nullptr == min || nullptr == max)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::UNIF;
	if (Identifier* parent = ordered_parent({min->get_uid(), max->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({min, max}, opcode, sample_grad);
}

Identifier* normal_sample (Identifier* mean, Identifier* stdev)
{
	if (nullptr == mean || nullptr == stdev)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::NORM;
	if (Identifier* parent = ordered_parent({mean->get_uid(), stdev->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({mean, stdev}, opcode, sample_grad);
}

Identifier* transpose (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::TRANSPOSE;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* transpose (Identifier* a, Identifier* perm)
{
	if (nullptr == a || nullptr == perm)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::TRANSPOSE;
	if (Identifier* parent = ordered_parent({a->get_uid(), perm->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, perm}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* transpose (Identifier* a, std::vector<uint64_t> perm)
{
	return transpose(a, Constant::get(perm, clay::Shape({perm.size()})));
}

Identifier* flip (Identifier* a, Identifier* dims)
{
	if (nullptr == a || nullptr == dims)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::FLIP;
	if (Identifier* parent = ordered_parent({a->get_uid(), dims->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, dims}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* flip (Identifier* a, std::vector<uint64_t> dims)
{
	return flip(a, Constant::get(dims));
}

Identifier* arg_max (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ARGMAX;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		throw std::bad_function_call();
	});
}

Identifier* arg_max (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ARGMAX;
	if (Identifier* parent = ordered_parent({a->get_uid(), dim->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, dim}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		throw std::bad_function_call();
	});
}

Identifier* arg_max (Identifier* a, uint64_t dim)
{
	return arg_max(a, Constant::get(dim));
}

Identifier* reduce_max (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RMAX;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		auto a = args.front();
		auto me = reduce_max(a);
		return mul(a->derive(wrt), eq(me, a));
	});
}

Identifier* reduce_max (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RMAX;
	if (Identifier* parent = ordered_parent({a->get_uid(), dim->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, dim}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		auto a = args.front();
		auto dim = args.back();
		auto me = reduce_max(a, dim);
		auto bitmap = expand(me, n_dimension(a, dim), dim);
		return mul(a->derive(wrt), eq(bitmap, a));
	});
}

Identifier* reduce_max (Identifier* a, uint64_t dim)
{
	return reduce_max(a, Constant::get(dim));
}

Identifier* reduce_sum (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RSUM;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* reduce_sum (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RSUM;
	if (Identifier* parent = ordered_parent({a->get_uid(), dim->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, dim}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* reduce_sum (Identifier* a, uint64_t dim)
{
	return reduce_sum(a, Constant::get(dim));
}

Identifier* reduce_mean (Identifier* a)
{
	auto denom = cast(a, n_elems(a));
	return div(reduce_sum(a), denom);
}

Identifier* reduce_mean (Identifier* a, Identifier* dim)
{
	auto denom = cast(a, n_dimension(a, dim));
	return div(reduce_sum(a, dim), denom);
}

Identifier* reduce_mean (Identifier* a, uint64_t dim)
{
	auto denom = cast(a, n_dimension(a, dim));
	return div(reduce_sum(a, dim), denom);
}

Identifier* reduce_l2norm (Identifier* a)
{
	return sqrt(reduce_sum(mul(a, a)));
}

Identifier* reduce_l2norm (Identifier* a, Identifier* dim)
{
	return sqrt(reduce_sum(mul(a, a), dim));
}

Identifier* reduce_l2norm (Identifier* a, uint64_t dim)
{
	return sqrt(reduce_sum(mul(a, a), dim));
}

Identifier* n_elems (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::N_ELEMS;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		Identifier* a = args.front();
		return make_zero(a);
	});
}

Identifier* n_dimension (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::N_DIMS;
	if (Identifier* parent = ordered_parent({a->get_uid(), dim->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, dim}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		Identifier* a = args.front();
		return make_zero(a);
	});
}

Identifier* n_dimension (Identifier* a, uint64_t dim)
{
	return n_dimension(a, Constant::get(dim));
}

Identifier* expand (Identifier* a, Identifier* n, Identifier* dim)
{
	if (nullptr == a || nullptr == n || nullptr == dim)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::EXPAND;
	if (Identifier* parent = ordered_parent({a->get_uid(), n->get_uid(), dim->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, n, dim}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* expand (Identifier* a, Identifier* n, uint64_t dim)
{
	return expand(a, n, Constant::get(dim));
}

Identifier* expand (Identifier* a, uint64_t n, uint64_t dim)
{
	return expand(a, Constant::get(n), Constant::get(dim));
}

Identifier* clip (Identifier* a, Identifier* min, Identifier* max)
{
	auto lt_min = lt(a, min);
	auto gt_max = gt(a, max);
	auto abetween = mul(mul(logical_not(lt_min), logical_not(gt_max)), a);
	return add(add(abetween, mul(lt_min, min)), mul(gt_max, max));
}

Identifier* clip_norm (Identifier* a, Identifier* cap)
{
	auto l2 = reduce_l2norm(a);
	auto is_clip = lt(l2, cap);
	auto no_clip = logical_not(is_clip);
	auto cli = div(mul(a, cap), l2);
	return add(mul(is_clip, cli), mul(no_clip, a));
}

Identifier* matmul (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::MATMUL;
	if (Identifier* parent = ordered_parent({a->get_uid(), b->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return nullptr;
	});
}

}

#endif
