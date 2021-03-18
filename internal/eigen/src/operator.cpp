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

EigenptrT project (const teq::TensptrT& in, const marsh::iAttributed& attrib)
{
	StorageIndicesT indices;
	Packer<StorageIndicesT>().unpack(indices, attrib);
	auto n = in->shape().n_elems();
	auto it = indices.begin();
	StorageIndicesT inner(it, it + n);
	StorageIndicesT outer(it + n, indices.end());
	return std::make_shared<ProjectOp>(*in, inner, outer);
}

numbers::Fraction matmul_density (
    const numbers::Fraction& ldensity, const numbers::Fraction& rdensity,
	teq::DimT common_dim)
{
    return reverse(pow(reverse(ldensity * rdensity), common_dim));
}

}

#endif
