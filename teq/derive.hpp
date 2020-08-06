///
/// derive.hpp
/// teq
///
/// Purpose:
/// Define interface for building derivative graphs
///

#include <list>

#include "teq/traveler.hpp"

#ifndef TEQ_DERIVE_HPP
#define TEQ_DERIVE_HPP

namespace teq
{

/// Interface for defining required methods for derivation
struct iDerivativeFuncs
{
	virtual ~iDerivativeFuncs (void) = default;

	/// Let op be functor F with arguments args,
	/// X is the ith argument of F, and supgrad be defined as dG/dF
	/// where G is some super-functor using F
	/// Return derivative G wrt to arg X by applying chain rule
	virtual TensptrT lderive (FuncptrT op,
		TensptrT supgrad, size_t i) const = 0;

	/// Return tensor representing 1 constant
	virtual TensptrT get_const_one (Shape shape) const = 0;

	/// Return tensor representing 0 constant
	virtual TensptrT get_const_zero (Shape shape) const = 0;

	/// Return functor representing sum(elems)
	virtual TensptrT add (TensptrsT elems) const = 0;
};

using GradMapT = TensMapT<TensptrsT>;

/// Define manditory definitions required for tensor differentiation
/// For some graph F(G(x)), chain rule for calculating dF/dx is
/// defined in the following order:
/// 1. calculate dF/dG => F local derivative and
/// derivative of super composition (supcomp_grad for G)
/// 2. calculate dG/dx => G local derivative
/// 3. chain dF/dG (supcomp_grad) and dG/dx (local_der)
/// This top-down approach updates tensor shape information such
/// that output derivative dF/dx has the shape of x
/// Return derivative of root with respect to target
TensptrsT derive (
	TensptrT root,
	const TensptrsT& targets,
	const iDerivativeFuncs& funcs);

/// Given derivatives of root R wrt to functors F in input grads,
/// where functors F are ancestors of parent root nodes,
/// continue looking for derivatives of R wrt to input targets and add to grads.
/// Derivatives are found using back-propagation.
void partial_derive (GradMapT& grads,
	const TensptrSetT& parents,
	const TensSetT& targets,
	const iDerivativeFuncs& funcs);

}

#endif // TEQ_DERIVE_HPP
