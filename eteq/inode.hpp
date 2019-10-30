//
/// inode.hpp
/// eteq
///
/// Purpose:
/// Define node interface and registration and conversion objects
///

#include "estd/estd.hpp"

#include "teq/itensor.hpp"

#include "eteq/eigen.hpp"

#ifndef ETEQ_INODE_HPP
#define ETEQ_INODE_HPP

namespace eteq
{

/// Interface node for wrapping typed tensor
template <typename T>
struct iNode
{
	static_assert(egen::TypeInfo<T>::type != egen::BAD_TYPE,
		"Cannot create node of unknown type");

	virtual ~iNode (void) = default;

	/// Return deep copy of node where internal typed tensor is copied
	iNode<T>* clone (void) const
	{
		return this->clone_impl();
	}

	/// Return shape of internal tensor
	teq::Shape shape (void)
	{
		return get_tensor()->shape();
	}

	/// Return string representation of internal tensor
	std::string to_string (void) const
	{
		return get_tensor()->to_string();
	}

	/// Return raw data stored in internal typed tensor
	virtual T* data (void) = 0;

	/// Trigger internal typed tensor update
	virtual void update (void) = 0;

	/// Return internal tensor
	virtual teq::TensptrT get_tensor (void) const = 0;

protected:
	virtual iNode<T>* clone_impl (void) const = 0;
};

/// Smart pointer of node
template <typename T>
using NodeptrT = std::shared_ptr<iNode<T>>;

/// Vector of nodes
template <typename T>
using NodesT = std::vector<NodeptrT<T>>;

/// Function for building a node from tensor
template <typename T>
using NodeBuilderF = std::function<NodeptrT<T>(teq::TensptrT)>;

/// Node registry of tensor types and tensor to node function
template <typename T>
struct NodeConverters final
{
	/// Return node associated with tensor type
	NodeptrT<T> to_node (teq::TensptrT tens)
	{
		if (nullptr == tens)
		{
			return nullptr;
		}
		const std::type_info& tp = typeid(*tens);
		return estd::must_getf(builders_, tp.hash_code(),
			"unknown tensor type `%s` with `%s` dtype",
			tp.name(), egen::name_type(egen::get_type<T>()).c_str())(tens);
	}

	/// Map tensor type to node creation function
	std::unordered_map<size_t,NodeBuilderF<T>> builders_;
};

/// Return global node converter of specified type
template <typename T>
NodeConverters<T>& get_converter (void)
{
	static NodeConverters<T> converter;
	return converter;
}

/// Return node of tens according to builders in specified converter
template <typename T>
NodeptrT<T> to_node (teq::TensptrT tens,
	NodeConverters<T>& converter = get_converter<T>())
{
	return converter.to_node(tens);
}

/// Return true if tensor type successfully registers and maps to node builder,
///	otherwise false
template <typename TensType, typename T>
bool register_builder (NodeBuilderF<T> builder)
{
	const std::type_info& tp = typeid(TensType);
	return get_converter<T>().builders_.
		emplace(tp.hash_code(), builder).second;
}

}

#endif // ETEQ_INODE_HPP
