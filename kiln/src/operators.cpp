//
//  operator.cpp
//  kiln
//

#include "kiln/operators.hpp"
#include "kiln/matmul_grad.hpp"

#ifdef KILN_OPERATORS_HPP

namespace kiln
{

static inline Identifier* single_parent (UID argid, slip::OPCODE opcode,
	mold::Range arange = {0,0})
{
	UIDRange arg{argid, arange};
	FunctorSetT funcs = Graph::get_global().get_func(opcode);
	for (Functor* f : funcs)
	{
		auto args = f->get_args();
		if (args.size() == 1 && args[0] == arg)
		{
			return f;
		}
	}
	return nullptr;
}

static inline Identifier* ordered_parent (
	std::vector<UIDRange> srcs, slip::OPCODE opcode)
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
	std::unordered_set<UIDRange,UIDRangeHasher> srcs, slip::OPCODE opcode)
{
	assert(srcs.size() > 0);
	FunctorSetT funcs = Graph::get_global().get_func(opcode);
	for (Functor* f : funcs)
	{
		auto args = f->get_args();
		std::unordered_set<UIDRange,UIDRangeHasher> argset(
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
	return args.front().second.arg_;
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
		return cast(args.front().first.arg_, args.back().second.arg_);
	}},
	std::pair<slip::OPCODE,GradF>{slip::ABS,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return abs(args.front().second.arg_);
	}},
	std::pair<slip::OPCODE,GradF>{slip::NEG,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return neg(args.front().second.arg_);
	}},
	std::pair<slip::OPCODE,GradF>{slip::NOT,
	[](Identifier* one, GradArgsT args) -> Identifier*
	{
		auto df = args.front().second.arg_;
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
		auto f = args.front().first.arg_;
		auto df = args.front().second.arg_;
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
		auto f = args.front().first.arg_;
		auto df = args.front().second.arg_;
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
		auto f = args.front().first.arg_;
		auto df = args.front().second.arg_;
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
		auto f = args.front().first.arg_;
		auto df = args.front().second.arg_;
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
		auto f = args.front().first.arg_;
		auto df = args.front().second.arg_;
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
		auto f = args.front().first.arg_;
		auto df = args.front().second.arg_;
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
		return round(args.front().second.arg_);
	}},
	std::pair<slip::OPCODE,GradF>{slip::ISMAX,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return mul(
			IdRange{is_max(args.front().first), mold::Range(0, 0)},
			IdRange{args.front().second.arg_, mold::Range(0, 0)});
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
		assert(g.arg_->has_data());
		Identifier* one = make_one(g.arg_);
		auto pre = pow(f, IdRange{
			sub(g, IdRange{one, mold::Range(0, 0)}),
			g.drange_
		});
		Identifier* lhs = nullptr;
		Identifier* rhs = nullptr;
		if (nullptr != df.arg_)
		{
			lhs = mul(df, g);
		}
		if (nullptr != dg.arg_)
		{
			rhs = mul(dg, IdRange{mul(f,
				IdRange{log(f.arg_), f.drange_}),
				f.drange_
			});
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
		if (nullptr == lhs.arg_)
		{
			return rhs.arg_;
		}
		else if (nullptr == rhs.arg_)
		{
			return lhs.arg_;
		}
		return add(lhs, rhs);
	}},
	std::pair<slip::OPCODE,GradF>{slip::SUB,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		// h'(f, g) = f' - g'
		auto lhs = args.front().second;
		auto rhs = args.back().second;
		if (nullptr == lhs.arg_)
		{
			return neg(rhs.arg_);
		}
		else if (nullptr == rhs.arg_)
		{
			return lhs.arg_;
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
		auto rhs = div(IdRange{div(
			IdRange{mul(dg, f), dg.drange_}, g), dg.drange_
		}, g);
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
		if (da.arg_ == nullptr)
		{
			return nullptr;
		}
		auto out = is_max(a);
		auto occ = reduce_sum(IdRange{out, a.drange_});
		auto g = div(IdRange{out, a.drange_}, IdRange{occ, mold::Range(0, 0)});
		return mul(IdRange{g, a.drange_}, da);
	}},
	std::pair<slip::OPCODE,GradF>{slip::RSUM, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::N_ELEMS, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::N_DIMS, zero_grad},
	std::pair<slip::OPCODE,GradF>{slip::EXPAND, straight_grad},
	std::pair<slip::OPCODE,GradF>{slip::MATMUL, matmul_grad},
	std::pair<slip::OPCODE,GradF>{slip::RESHAPE,
	[](Identifier*, GradArgsT args) -> Identifier*
	{
		return reshape(args.front().second.arg_, args.back().first.arg_);
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
	if (Identifier* parent = ordered_parent({
		{type->get_uid(), mold::Range(0, 0)},
    	{a->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{type, a}, opcode);
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

Identifier* flip (Identifier* a, Identifier* dims)
{
	if (nullptr == a || nullptr == dims)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::FLIP;
	if (Identifier* parent = ordered_parent({
		{a->get_uid(), mold::Range(0, 0)},
		{dims->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{a, dims}, opcode);
}

Identifier* flip (Identifier* a, std::vector<uint64_t> dims)
{
	Identifier* did = Constant::get(dims);
	assoc(a, did);
	return flip(a, did);
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
	if (Identifier* parent = ordered_parent({
		{a->get_uid(), mold::Range(0, 0)},
		{perm->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{a, perm}, opcode);
}

Identifier* transpose (Identifier* a, std::vector<uint64_t> perm)
{
	Identifier* pid = Constant::get(perm, clay::Shape({perm.size()}));
	assoc(a, pid);
	return transpose(a, pid);
}

Identifier* expand (Identifier* a, uint64_t n, uint64_t dim)
{
	return expand(IdRange{a, mold::Range(dim, dim)}, n);
}

Identifier* expand (IdRange a, uint64_t n)
{
	if (nullptr == a.arg_)
	{
		return nullptr;
	}
	Identifier* nid = Constant::get(n);
	assoc(a.arg_, nid);
	IdRange nrange{nid, mold::Range(0, 0)};

	slip::OPCODE opcode = slip::EXPAND;
	if (Identifier* parent = ordered_parent({
		a.get_uid(), nrange.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, nrange}, opcode);
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

// dimensioned operators

Identifier* pow (Identifier* b, Identifier* x)
{
	return pow(IdRange{b, mold::Range(0, 0)}, IdRange{x, mold::Range(0, 0)});
}

Identifier* add (Identifier* a, Identifier* b)
{
	return add(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* sub (Identifier* a, Identifier* b)
{
	return sub(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* mul (Identifier* a, Identifier* b)
{
	return mul(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* div (Identifier* a, Identifier* b)
{
	return div(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* eq (Identifier* a, Identifier* b)
{
	return eq(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* neq (Identifier* a, Identifier* b)
{
	return neq(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* lt (Identifier* a, Identifier* b)
{
	return lt(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* gt (Identifier* a, Identifier* b)
{
	return gt(IdRange{a, mold::Range(0, 0)}, IdRange{b, mold::Range(0, 0)});
}

Identifier* matmul (Identifier* a, Identifier* b)
{
	return matmul(IdRange{a, mold::Range(0, 2)}, IdRange{b, mold::Range(0, 2)});
}


Identifier* binomial_sample (Identifier* n, Identifier* p)
{
	if (nullptr == n || nullptr == p)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::BINO;
	if (Identifier* parent = ordered_parent({
		{n->get_uid(), mold::Range(0, 0)},
		{p->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{n, p}, opcode);
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
	if (Identifier* parent = ordered_parent({
		{min->get_uid(), mold::Range(0, 0)},
		{max->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{min, max}, opcode);
}

Identifier* normal_sample (Identifier* mean, Identifier* stdev)
{
	if (nullptr == mean || nullptr == stdev)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::NORM;
	if (Identifier* parent = ordered_parent({
		{mean->get_uid(), mold::Range(0, 0)},
		{stdev->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{mean, stdev}, opcode);
}


Identifier* n_dimension (Identifier* a, uint64_t dim)
{
	return n_dimension(IdRange{a, mold::Range(dim, dim)});
}

Identifier* arg_max (Identifier* a)
{
	return arg_max(IdRange{a, mold::Range(0, -1)});
}

Identifier* arg_max (Identifier* a, uint64_t dim)
{
	return arg_max(IdRange{a, mold::Range(dim, dim + 1)});
}

Identifier* reduce_max (Identifier* a)
{
	return reduce_max(IdRange{a, mold::Range(0, -1)});
}

Identifier* reduce_max (Identifier* a, uint64_t dim)
{
	return reduce_max(IdRange{a, mold::Range(dim, dim + 1)});
}

Identifier* reduce_sum (Identifier* a)
{
	return reduce_sum(IdRange{a, mold::Range(0, -1)});
}

Identifier* reduce_sum (Identifier* a, uint64_t dim)
{
	return reduce_sum(IdRange{a, mold::Range(dim, dim + 1)});
}



Identifier* pow (IdRange b, IdRange x)
{
	if (nullptr == b.arg_ || nullptr == x.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::POW;
	if (Identifier* parent = ordered_parent({
		b.get_uid(), x.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({b, x}, opcode);
}

Identifier* add (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ADD;
	if (Identifier* parent = unordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* sub (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::SUB;
	if (Identifier* parent = ordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* mul (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::MUL;
	if (Identifier* parent = unordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* div (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::DIV;
	if (Identifier* parent = ordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* eq (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::EQ;
	if (Identifier* parent = unordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* neq (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::NE;
	if (Identifier* parent = unordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* lt (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::LT;
	if (Identifier* parent = ordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* gt (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::GT;
	if (Identifier* parent = ordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}

Identifier* matmul (IdRange a, IdRange b)
{
	if (nullptr == a.arg_ || nullptr == b.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::MATMUL;
	if (Identifier* parent = ordered_parent({
		a.get_uid(), b.get_uid()}, opcode))
	{
		return parent;
	}
	return new Functor({a, b}, opcode);
}


Identifier* n_dimension (IdRange a)
{
	if (nullptr == a.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::N_DIMS;
	if (Identifier* parent = single_parent(a.arg_->get_uid(), opcode, a.drange_))
	{
		return parent;
	}
	return new Functor({a}, opcode);
}

Identifier* arg_max (IdRange a)
{
	if (nullptr == a.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ARGMAX; // todo: optimize include dim
	if (Identifier* parent = single_parent(a.arg_->get_uid(), opcode, a.drange_))
	{
		return parent;
	}
	return new Functor({a}, opcode);
}

Identifier* is_max (IdRange a)
{
	if (nullptr == a.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::ISMAX; // todo: optimize include dim
	if (Identifier* parent = single_parent(a.arg_->get_uid(), opcode, a.drange_))
	{
		return parent;
	}
	return new Functor({a}, opcode);
}

Identifier* reduce_max (IdRange a)
{
	if (nullptr == a.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RMAX; // todo: optimize include dim
	if (Identifier* parent = single_parent(a.arg_->get_uid(), opcode, a.drange_))
	{
		return parent;
	}
	return new Functor({a}, opcode);
}

Identifier* reduce_sum (IdRange a)
{
	if (nullptr == a.arg_)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RSUM; // todo: optimize include dim
	if (Identifier* parent = single_parent(a.arg_->get_uid(), opcode, a.drange_))
	{
		return parent;
	}
	return new Functor({a}, opcode);
}



Identifier* reduce_mean (Identifier* a)
{
	auto denom = cast(a, n_elems(a));
	return div(reduce_sum(a), denom);
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

Identifier* reduce_l2norm (Identifier* a, uint64_t dim)
{
	return sqrt(reduce_sum(mul(a, a), dim));
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

Identifier* reshape (Identifier* a, Identifier* shape)
{
	if (nullptr == a || nullptr == shape)
	{
		return nullptr;
	}
	slip::OPCODE opcode = slip::RESHAPE;
	if (Identifier* parent = ordered_parent({
		{a->get_uid(), mold::Range(0, 0)},
		{shape->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{a, shape}, opcode);
}

Identifier* reshape (Identifier* a, std::vector<uint64_t> shape)
{
	Identifier* sid = Constant::get(shape,
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
	if (Identifier* parent = ordered_parent({
		{a->get_uid(), mold::Range(0, 0)},
		{b->get_uid(), mold::Range(0, 0)},
		{dims->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor({a, b, dims}, opcode);
}

Identifier* jacobian (Identifier* a, Identifier* b, uint64_t targetdim, uint64_t swapdim)
{
	Identifier* did = Constant::get(std::vector<uint64_t>{
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
	if (Identifier* parent = ordered_parent({
		{a->get_uid(), mold::Range(0, 0)},
		{dim->get_uid(), mold::Range(0, 0)}}, opcode))
	{
		return parent;
	}
	return new Functor(std::vector<Identifier*>{a, dim}, opcode);
}

Identifier* trace_expand (Identifier* a, uint64_t dim)
{
	Identifier* did = Constant::get(dim);
	assoc(a, did);
	return trace_expand(a, did);
}

}

#endif
