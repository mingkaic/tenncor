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

}

#endif
