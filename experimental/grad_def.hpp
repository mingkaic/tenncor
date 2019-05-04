#include "ade/itensor.hpp"

#ifndef ADE_GRAD_DEF_HPP
#define ADE_GRAD_DEF_HPP

namespace ade
{

/// Define manditory definitions required for tensor differentiation
/// For some graph F(G(x)), chain rule for calculating dF/dx is
/// defined in the following order:
///     1. calcualte dF/dG => F local derivative and
///        derivative of super composition (supcomp_grad for G)
///     2. calculate dG/dx => G local derivative
///     3. chain dF/dG (supcomp_grad) and dG/dx (local_der)
/// This top-down approach updates tensor shape information such
/// that output derivative dF/dx has the shape of x
struct iDeriver
{
	virtual ~iDeriver (void) = default;

	/// Let op be functor F with arguments args
	/// Return derivative of F wrt args[arg_idx]
	virtual TensptrT local_derivative (std::shared_ptr<iFunctor> op,
		size_t arg_idx) const = 0;

	/// Let op be functor F with arguments args, and
	///     local_der is derivative of F wrt one of args (say x)
	/// Let supcomp_grad be defined as dG/dF
	///     where G is some super-functor using F
	/// Return derivative G wrt to arg x by applying chain rule
	virtual TensptrT chain_rule (std::shared_ptr<iFunctor> op,
		const TensptrT& local_der, TensptrT supcomp_grad) = 0;
};

}

#endif // ADE_GRAD_DEF_HPP
