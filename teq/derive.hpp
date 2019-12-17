///
/// grad_def.hpp
/// teq
///
/// Purpose:
/// Define interface for building derivative graphs
///

#include "teq/traveler.hpp"

#ifndef TEQ_DERIVE_HPP
#define TEQ_DERIVE_HPP

namespace teq
{

/// Interface for defining required methods for derivation
struct iDerivativeFuncs
{
	/// Let op be functor F with arguments args
	/// Return derivative of F wrt args[arg_idx]
	virtual TensptrT local_derivative (FuncptrT op, size_t arg_idx) const = 0;

	/// Let op be functor F with arguments args, and
	/// local_der is derivative of F wrt one of args (say x)
	/// Let supcomp_grad be defined as dG/dF
	/// where G is some super-functor using F
	/// Return derivative G wrt to arg x by applying chain rule
	virtual TensptrT chain_rule (FuncptrT op, const TensptrT& local_der,
		TensptrT supcomp_grad, size_t arg_idx) const = 0;

	/// Return tensor representing 1 constant
	virtual TensptrT get_const_one (Shape shape) const = 0;

	/// Return tensor representing 0 constant
	virtual TensptrT get_const_zero (Shape shape) const = 0;

	/// Return functor representing sum(elems)
	virtual TensptrT add (TensptrsT elems) const = 0;
};

/// Define manditory definitions required for tensor differentiation
/// For some graph F(G(x)), chain rule for calculating dF/dx is
/// defined in the following order:
/// 1. calcualte dF/dG => F local derivative and
/// derivative of super composition (supcomp_grad for G)
/// 2. calculate dG/dx => G local derivative
/// 3. chain dF/dG (supcomp_grad) and dG/dx (local_der)
/// This top-down approach updates tensor shape information such
/// that output derivative dF/dx has the shape of x
/// Return derivative of root with respect to target
TensptrT derive (TensptrT root, TensptrT target, iDerivativeFuncs& funcs);

}

#endif // TEQ_DERIVE_HPP
