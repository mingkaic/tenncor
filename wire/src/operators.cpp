//
//  operator.cpp
//  wire
//

#include "wire/operators.hpp"
#include "wire/matmul_grad.hpp"

#ifdef WIRE_OPERATORS_HPP

namespace wire
{

static inline Identifier* single_parent (UID argid, slip::OPCODE opcode)
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
	std::vector<UID> srcs, slip::OPCODE opcode)
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
	std::unordered_set<UID> srcs, slip::OPCODE opcode)
{
	assert(srcs.size() > 0);
	FunctorSetT funcs = Graph::get_global().get_func(opcode);
	for (Functor* f : funcs)
	{
		auto args = f->get_args();
		std::unordered_set<UID> argset(
			args.begin(), args.end());
		if (argset == srcs)
		{
			return f;
		}
	}
	return nullptr;
}

static Identifier* straight_grad (Identifier*, GradArgsT args)
{
	return args.front().second;
}

static Constant* zero_grad (Identifier*, GradArgsT args)
{
	return nullptr;
}

const GradMapT grad_op =
{
	std::pair<slip::OPCODE,GradF>{slip::CAST,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return cast(args.front().first, args.back().second);
	}},
	std::pair<slip::OPCODE,GradF>{slip::ABS,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return abs(args.front().second);
	}},
	std::pair<slip::OPCODE,GradF>{slip::NEG,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return neg(args.front().second);
	}},
	std::pair<slip::OPCODE,GradF>{slip::NOT,
	[](Identifier* one, GradArgsT args) -> Identifier*
	{
		auto df = args.front().second;
		if (nullptr == df)
		{
			return one;
		}
		return logical_not(df);
	}},
	std::pair<slip::OPCODE,GradF>{slip::SIN,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// sin'(f) = f'*cos(f)
		auto f = args.front().first;
		auto df = args.front().second;
		if (nullptr == df)
		{
			return nullptr;
		}
		return mul(df, cos(f));
	}},
	std::pair<slip::OPCODE,GradF>{slip::COS,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// cos'(f) = -f'*sin(f)
		auto f = args.front().first;
		auto df = args.front().second;
		if (nullptr == df)
		{
			return nullptr;
		}
		return mul(neg(df), sin(f));
	}},
	std::pair<slip::OPCODE,GradF>{slip::TAN,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// tan'(f) = f'*sec^2(f)
		// 		= f'/cos^2(f)
		auto f = args.front().first;
		auto df = args.front().second;
		if (nullptr == df)
		{
			return nullptr;
		}
		auto denom = cos(f);
		return div(df, mul(denom, denom));
	}},
	std::pair<slip::OPCODE,GradF>{slip::EXP,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// exp'(f) = f'*exp(f)
		auto f = args.front().first;
		auto df = args.front().second;
		if (nullptr == df)
		{
			return nullptr;
		}
		return mul(df, exp(f));
	}},
	std::pair<slip::OPCODE,GradF>{slip::LOG,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// log'(f) = f' / f
		auto f = args.front().first;
		auto df = args.front().second;
		if (nullptr == df)
		{
			return nullptr;
		}
		return div(df, f);
	}},
	std::pair<slip::OPCODE,GradF>{slip::SQRT,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// sqrt'(f) = f'/(2*sqrt(f))
		auto f = args.front().first;
		auto df = args.front().second;
		if (nullptr == df)
		{
			return nullptr;
		}
		auto denom = sqrt(f);
		return div(df, add(denom, denom));
	}},
	std::pair<slip::OPCODE,GradF>{slip::ROUND,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// round'(f) = round(f')
		return round(args.front().second);
	}},
	std::pair<slip::OPCODE,GradF>{slip::POW,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
		//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
		auto f = args.front().first;
		auto g = args.back().first;
		auto df = args.front().second;
		auto dg = args.back().second;
		assert(g->has_data());
		Identifier* one = make_one(g);
		auto pre = pow(f, sub(g, one));
		Identifier* lhs = nullptr;
		Identifier* rhs = nullptr;
		if (nullptr != df)
		{
			lhs = mul(df, g);
		}
		if (nullptr != dg)
		{
			rhs = mul(dg, mul(f, log(f)));
		}
		Identifier* main;
		if (nullptr == lhs)
		{
			main = rhs;
		}
		else if (nullptr == rhs)
		{
			main = lhs;
		}
		else
		{
			main = add(lhs, rhs);
		}
		return mul(pre, main);
	}},
	std::pair<slip::OPCODE,GradF>{slip::ADD,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// h'(f, g) = f' + g'
		auto lhs = args.front().second;
		auto rhs = args.back().second;
		if (nullptr == lhs)
		{
			return rhs;
		}
		else if (nullptr == rhs)
		{
			return lhs;
		}
		return add(lhs, rhs);
	}},
	std::pair<slip::OPCODE,GradF>{slip::SUB,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// h'(f, g) = f' - g'
		auto lhs = args.front().second;
		auto rhs = args.back().second;
		if (nullptr == lhs)
		{
			return neg(rhs);
		}
		else if (nullptr == rhs)
		{
			return lhs;
		}
		return sub(lhs, rhs);
	}},
	std::pair<slip::OPCODE,GradF>{slip::MUL,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// h'(f, g) = f' * g + g' * f
		auto f = args.front().first;
		auto df = args.front().second;
		auto g = args.back().first;
		auto dg = args.back().second;
		auto lhs = mul(df, g);
		auto rhs = mul(dg, f);
		if (nullptr == lhs)
		{
			return rhs;
		}
		else if (nullptr == rhs)
		{
			return lhs;
		}
		return add(lhs, rhs);
	}},
	std::pair<slip::OPCODE,GradF>{slip::DIV,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// h'(f, g) = (f' * g - g' * f) / g^2
		//			= f' / g - ((g' * f) / g) / g
		auto f = args.front().first;
		auto df = args.front().second;
		auto g = args.back().first;
		auto dg = args.back().second;
		auto lhs = div(df, g);
		auto rhs = div(div(mul(dg, f), g), g);
		if (nullptr == lhs)
		{
			return neg(rhs);
		}
		else if (nullptr == rhs)
		{
			return lhs;
		}
		return sub(lhs, rhs);
	}},
	std::pair<slip::OPCODE,GradF>{slip::EQ,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return eq(args.front().first, args.back().first);
	}},
	std::pair<slip::OPCODE,GradF>{slip::NE,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return neq(args.front().first, args.back().first);
	}},
	std::pair<slip::OPCODE,GradF>{slip::LT,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return lt(args.front().first, args.back().first);
	}},
	std::pair<slip::OPCODE,GradF>{slip::GT,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return gt(args.front().first, args.back().first);
	}},
	std::pair<slip::OPCODE,GradF>{slip::BINO, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::UNIF, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::NORM, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::TRANSPOSE, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::FLIP, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::UARGMAX,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		throw std::bad_function_call();
	}},
	std::pair<slip::OPCODE,GradF>{slip::URMAX,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		auto a = args.front().first;
		auto da = args.front().second;
		if (da == nullptr)
		{
			return nullptr;
		}
		auto me = reduce_max(a);
		return mul(da, eq(me, a));
	}},
	std::pair<slip::OPCODE,GradF>{slip::URSUM, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::ARGMAX,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		throw std::bad_function_call();
	}},
	std::pair<slip::OPCODE,GradF>{slip::RMAX,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		auto a = args.front().first;
		auto da = args.front().second;
		if (da == nullptr)
		{
			return nullptr;
		}
		auto dim = args.back().first;
		auto me = reduce_max(a, dim);
		auto bitmap = expand(me, n_dimension(a, dim), dim);
		return mul(da, eq(bitmap, a));
	}},
	std::pair<slip::OPCODE,GradF>{slip::RSUM, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::N_ELEMS, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::N_DIMS, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::EXPAND, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::MATMUL, matmul_grad},
	std::pair<slip::OPCODE,GradF>{slip::RESHAPE,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return reshape(args.front().second, args.back().first);
	}},
	std::pair<slip::OPCODE,GradF>{slip::JACOBIAN,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		throw std::bad_function_call(); // todo: implement
	}}
};

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
	return new Functor({type, a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({b, x}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({a, b}, opcode);
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
	return new Functor({n, p}, opcode);
}

Identifier* binomial_sample (Identifier* n, double p)
{
	Identifier* pid = Constant::get(p);
	assoc(n, pid);
	return binomial_sample(n, pid);
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
	return new Functor({min, max}, opcode);
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
	return new Functor({mean, stdev}, opcode);
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
	return new Functor({a}, opcode);
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
	return new Functor({a, perm}, opcode);
}

Identifier* transpose (Identifier* a, std::vector<uint64_t> perm)
{
	Identifier* pid = Constant::get(perm, clay::Shape({perm.size()}));
	assoc(a, pid);
	return transpose(a, pid);
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
	return new Functor({a, dims}, opcode);
}

Identifier* flip (Identifier* a, std::vector<uint64_t> dims)
{
	Identifier* did = Constant::get(dims);
	assoc(a, did);
	return flip(a, did);
}

Identifier* arg_max (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::UARGMAX;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode);
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
	return new Functor({a, dim}, opcode);
}

Identifier* arg_max (Identifier* a, uint64_t dim)
{
	Identifier* did = Constant::get(dim);
	assoc(a, did);
	return arg_max(a, did);
}

Identifier* reduce_max (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::URMAX;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode);
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
	return new Functor({a, dim}, opcode);
}

Identifier* reduce_max (Identifier* a, uint64_t dim)
{
	Identifier* did = Constant::get(dim);
	assoc(a, did);
	return reduce_max(a, did);
}

Identifier* reduce_sum (Identifier* a)
{
	if (nullptr == a)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::URSUM;
	if (Identifier* parent = single_parent(a->get_uid(), opcode))
	{
		return parent;
	}
	return new Functor({a}, opcode);
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
	return new Functor({a, dim}, opcode);
}

Identifier* reduce_sum (Identifier* a, uint64_t dim)
{
	Identifier* did = Constant::get(dim);
	assoc(a, did);
	return reduce_sum(a, did);
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
	return new Functor({a}, opcode);
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
	return new Functor({a, dim}, opcode);
}

Identifier* n_dimension (Identifier* a, uint64_t dim)
{
	Identifier* did = Constant::get(dim);
	assoc(a, did);
	return n_dimension(a, did);
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
	return new Functor({a, n, dim}, opcode);
}

Identifier* expand (Identifier* a, Identifier* n, uint64_t dim)
{
	Identifier* did = Constant::get(dim);
	assoc(a, did);
	return expand(a, n, did);
}

Identifier* expand (Identifier* a, uint64_t n, uint64_t dim)
{
	Identifier* nid = Constant::get(n);
	assoc(a, nid);
	return expand(a, nid, dim);
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
	return new Functor({a, b}, opcode);
}

Identifier* reshape (Identifier* a, Identifier* shape)
{
	if (nullptr == a || nullptr == shape)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RESHAPE;
	if (Identifier* parent = ordered_parent({a->get_uid(), shape->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, shape}, opcode);
}

Identifier* reshape (Identifier* a, std::vector<uint64_t> shape)
{
	Identifier* sid = wire::Constant::get(shape,
        clay::Shape({shape.size()}));
	assoc(a, sid);
    return reshape(a, sid);
}

Identifier* jacobian (Identifier* a, Identifier* b, Identifier* dims)
{
	if (nullptr == a || nullptr == b || nullptr == dims)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::JACOBIAN;
	if (Identifier* parent = ordered_parent(
		{a->get_uid(), b->get_uid(), dims->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b, dims}, opcode);
}

Identifier* jacobian (Identifier* a, Identifier* b, uint64_t targetdim, uint64_t swapdim)
{
	Identifier* did = wire::Constant::get(std::vector<uint64_t>{
		targetdim, swapdim}, clay::Shape({2}));
	assoc(a, did);
	return jacobian(a, b, did);
}

Identifier* trace_expand (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::TRACE_EXPAND;
	if (Identifier* parent = ordered_parent(
		{a->get_uid(), dim->get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, dim}, opcode);
}

Identifier* trace_expand (Identifier* a, uint64_t dim)
{
	Identifier* did = wire::Constant::get(dim);
	assoc(a, did);
	return trace_expand(a, did);
}

}

#endif
