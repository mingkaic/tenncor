#ifdef PERM_OP
#include "internal/eigen/perm_operator.hpp"
#else
#include "internal/eigen/operator.hpp"
#endif

#ifdef EIGEN_OPERATOR_HPP

namespace eigen
{

EigenptrT ref (const teq::TensptrT& in)
{
	return std::make_shared<TensRef>(*in);
}

}

#endif
