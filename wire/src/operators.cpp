//
//  operator.cpp
//  wire
//

#include "wire/operators.hpp"

#ifdef WIRE_OPERATORS_HPP

namespace wire
{

void assert_type (Identifier* a, TENS_TYPE type)
{}

void assert_shape (Identifier* a, tshape shape)
{}

Identifier* abs (Identifier* a)
{
	if (nullptr == a) return nullptr;
    return new Functor({a}, slip::ABS,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return abs(args.front()->derive(wrt));
	});
}

Identifier* neg (Identifier* a)
{
	if (nullptr == a) return nullptr;
    return new Functor({a}, slip::NEG,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return neg(args.front()->derive(wrt));
	});
}

Identifier* logical_not (Identifier* a)
{
	if (nullptr == a) return nullptr;
    return new Functor({a}, slip::NOT,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return logical_not(args.front()->derive(wrt));
	});
}

Identifier* sin (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::SIN,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// sin'(f) = f'*cos(f)
        auto f = args.front();
		return mul(f->derive(wrt), cos(f));
	});
}

Identifier* cos (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::COS,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// cos'(f) = -f'*sin(f)
        auto f = args.front();
		return mul(neg(f->derive(wrt)), sin(f));
	});
}

Identifier* tan (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::TAN,
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
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::EXP,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// exp'(f) = f'*exp(f)
        auto f = args.front();
		return mul(f->derive(wrt), exp(f));
	});
}

Identifier* log (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::LOG,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// log'(f) = f' / f
        auto f = args.front();
		return div(f->derive(wrt), f);
	});
}

Identifier* sqrt (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::SQRT,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// sqrt'(f) = f'/(2*sqrt(f))
        auto f = args.front();
        auto denom = sqrt(f);
		return div(f->derive(wrt), sum(f, f));
	});
}

Identifier* round (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::ROUND,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// round'(f) = round(f')
		return round(args.front()->derive(wrt));
	});
}

Identifier* pow (Identifier* b, Identifier* x)
{
	if (nullptr == b || nullptr == x) return nullptr;
	return new Functor({a, b}, slip::POW,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
		//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
        auto f = args.front();
        auto g = args.back();
		assert(g->has_data());
		auto lhs = pow(f, sub(g, make_one(g->get_state().dtype_)));
		auto rhs = add(mul(f->derive(wrt), g), mul(g->derive(wrt), mul(f, log(f))));
		return mul(lhs, rhs);
	});
}

Identifier* add (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::ADD,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = f' + g'
		return add(args.front()->derive(wrt), args.back()->derive(wrt));
	});
}

Identifier* sub (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::SUB,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = f' - g'
		return sub(args.front()->derive(wrt), args.back()->derive(wrt));
	});
}

Identifier* mul (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::MUL,
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
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::DIV,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		// h'(f, g) = (f' * g - g' * f) / g^2
		//		    = (f' / g - (g' * f) / g) / g
        auto f = args.front();
        auto g = args.back();
		auto num = sub(div(f->derive(wrt), g), div(mul(g->derive(wrt), f), g));
		return div(num, g);
	});
}

Identifier* eq (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::EQ,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return eq(args.front(), args.end());
	});
}

Identifier* neq (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::NE,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return neq(args.front(), args.end());
	});
}

Identifier* lt (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::LT,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return lt(args.front(), args.end());
	});
}

Identifier* gt (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::GT,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return gt(args.front(), args.end());
	});
}

Constant* sample_grad (Identifier*, std::vector<Identifier*> args)
{
	clay::State state = args.front().get_state();
	unsigned short nbytes = clay::type_size(state.dtype_) *
		state.shape_.n_elems();
	std::shared_ptr<char> data = clay::make_char(nbytes);
	memset(data.get(), 0, nbytes);
	return new Constant(data, state.shape_, nbytes.dtype_);
}

Identifier* binomial_sample (Identifier* n, Identifier* p)
{
	if (nullptr == n || nullptr == p) return nullptr;
	return new Functor({n, p}, slip::BINO, sample_grad);
}

Identifier* binomial_sample (Identifier* n, double p)
{
	return binomial_sample(n, Constant::get(p));
}

Identifier* uniform_sample (Identifier* min, Identifier* max)
{
	if (nullptr == min || nullptr == max) return nullptr;
	return new Functor({min, max}, slip::UNIF, sample_grad);
}

Identifier* normal_sample (Identifier* mean, Identifier* stdev)
{
	if (nullptr == mean || nullptr == stdev) return nullptr;
	return new Functor({mean, stdev}, slip::NORM, sample_grad);
}

Identifier* transpose (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::TRANSPOSE,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* transpose (Identifier* a, Identifier* perm)
{
	if (nullptr == a || nullptr == perm) return nullptr;
	return new Functor({a, perm}, slip::TRANSPOSE,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* transpose (Identifier* a, std::vector<uint64_t> perm)
{
	return transpose(a, Constant::get(perm));
}

Identifier* flip (Identifier* a, Identifier* dims)
{
	if (nullptr == a || nullptr == dims) return nullptr;
	return new Functor({a, dims}, slip::FLIP,
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
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::ARGMAX,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		throw std::exception();
	});
}

Identifier* arg_max (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim) return nullptr;
	return new Functor({a, dim}, slip::NOT,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		throw std::exception();
	});
}

Identifier* arg_max (Identifier* a, uint64_t dim)
{
	return arg_max(a, Constant::get(dim));
}

Identifier* reduce_max (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::RMAX,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		auto a = args.front();
		auto me = reduce_max(a);
		return mul(a->derive(wrt), eq(me, a));
	});
}

Identifier* reduce_max (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim) return nullptr;
	return new Functor({a, dim}, slip::NOT,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		auto a = args.front();
		auto dim = args.back();
		varptr me = reduce_max(a, dim);
		varptr bitmap = expand(me, n_dimension(a, dim), dim);
		return mul(a->derive(wrt), eq(bitmap, a));
	});
}

Identifier* reduce_max (Identifier* a, uint64_t dim)
{
	return reduce_max(a, Constant::get(dim));
}

Identifier* reduce_sum (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::RSUM,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return args.front()->derive(wrt);
	});
}

Identifier* reduce_sum (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim) return nullptr;
	return new Functor({a}, slip::NOT,
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
	return reduce_sum(a) / n_elems(a);
}

Identifier* reduce_mean (Identifier* a, Identifier* dim)
{
	return reduce_sum(a, dim) / n_dimension(a, dim);
}

Identifier* reduce_mean (Identifier* a, uint64_t dim)
{
	return reduce_sum(a, dim) / n_dimension(a, dim);
}

Identifier* reduce_l2norm (Identifier* a)
{
	return sqrt(reduce_sum(a * a));
}

Identifier* reduce_l2norm (Identifier* a, Identifier* dim)
{
	return sqrt(reduce_sum(a * a, dim));
}

Identifier* reduce_l2norm (Identifier* a, uint64_t dim)
{
	return sqrt(reduce_sum(a * a, dim));
}

Identifier* n_elems (Identifier* a)
{
	if (nullptr == a) return nullptr;
	return new Functor({a}, slip::N_ELEMS,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return nullptr;
	});
}

Identifier* n_dimension (Identifier* a, Identifier* dim)
{
	if (nullptr == a || nullptr == dim) return nullptr;
	return new Functor({a, dim}, slip::N_DIMS,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return nullptr;
	});
}

Identifier* n_dimension (Identifier* a, uint64_t dim)
{
	return n_dimensions(a, Constant::get(dim));
}

Identifier* expand (Identifier* a, Identifier* n, Identifier* dim)
{
	if (nullptr == a || nullptr == n || nullptr == dim) return nullptr;
	return new Functor({a, n, dim}, slip::EXPAND,
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
	assert_shape(cap, std::vector<size_t>{1});
	auto l2 = reduce_l2norm(a);
	auto is_clip = lt(l2, cap);
	auto no_clip = logical_not(is_clip);
	auto cli = div(mul(a, cap), l2);
	return add(mul(is_clip, cli), mul(no_clip, a));
}

Identifier* matmul (Identifier* a, Identifier* b)
{
	if (nullptr == a || nullptr == b) return nullptr;
	return new Functor({a, b}, slip::MATMUL,
	[](Identifier* wrt, std::vector<Identifier*> args) -> Identifier*
	{
		return nullptr;
	});
}

}

#endif
