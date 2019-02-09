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

	virtual void update (void) = 0;

	virtual TensMapT<T>* get_tensmap (void) = 0;

	virtual ade::TensptrT get_tensor (void) = 0;
};

template <typename T>
using NodeptrT = std::shared_ptr<iNode<T>>;

}

#endif // EAD_INODE_HPP
