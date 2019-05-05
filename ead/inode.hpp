#include "ade/itensor.hpp"

#include "ead/tensor.hpp"

#ifndef EAD_INODE_HPP
#define EAD_INODE_HPP

namespace ead
{

template <typename T>
struct iNode
{
	virtual ~iNode (void) = default;

	ade::Shape shape (void)
	{
		return get_tensor()->shape();
	}

	virtual T* data (void) = 0;

	virtual void update (void) = 0;

	virtual ade::TensptrT get_tensor (void) = 0;
};

template <typename T>
using NodeptrT = std::shared_ptr<iNode<T>>;

template <typename T>
using NodesT = std::vector<NodeptrT<T>>;

}

#endif // EAD_INODE_HPP
