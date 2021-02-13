///
/// derive.hpp
/// teq
///
/// Purpose:
/// Define interface for building derivative graphs
///

#ifndef TEQ_DERIVE_HPP
#define TEQ_DERIVE_HPP

#include <list>

#include "internal/teq/traveler.hpp"

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

	/// Return tensor representing 1 constant shaped like reference tensor
	virtual TensptrT get_const_one (iTensor& reference) const = 0;

	/// Return tensor representing 0 constant
	virtual TensptrT get_const_zero (iTensor& reference) const = 0;

	/// Return functor representing sum(elems)
	virtual TensptrT add (TensptrsT elems) const = 0;
};

struct iBackpropFuncs : public iDerivativeFuncs
{
	virtual ~iBackpropFuncs (void) = default;

	/// Let Jac(F,X) denote the Jacobian matrix
	/// mapping the derivative F wrt X.
	/// Return Jac(op,op.args[i]) @ prev_chain,
	/// assuming prev_chain is Jac(op.args[i],X).
	/// This function also handles optimization in cases
	/// when Jac(op,op.args[i]) is not needed
	/// Jacobian matrix takes shape <X.shape.n_elems,op.shape.n_elems>
	virtual TensptrT jacobian_chain (FuncptrT op,
		TensptrT prev_chain, size_t i) const = 0;

	virtual TensptrT dejacobianize (
		TensptrT jacobian, TensptrT x) const = 0;

	/// Return tensor representing shaped identity
	virtual TensptrT get_const_eye (NElemT n, size_t type_code) const = 0;
};

using JLinkMapT = TensMapT<TensMapT<TensptrsT>>;

using GradMapT = TensMapT<TensptrsT>;

/// Derive using backpropagation
TensptrsT backprop (
	TensptrT root,
	const TensptrsT& targets,
	const iBackpropFuncs& funcs);

void link_jacobian (JLinkMapT& jacobians,
	const TensptrSetT& parents, const TensSetT& targets,
	const iBackpropFuncs& funcs);

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
