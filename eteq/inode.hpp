#include "estd/estd.hpp"

#include "teq/itensor.hpp"

#include "eteq/eigen.hpp"

#ifndef ETEQ_INODE_HPP
#define ETEQ_INODE_HPP

namespace eteq
{

template <typename T>
struct iNode
{
	static_assert(egen::TypeInfo<T>::type != egen::BAD_TYPE,
		"Cannot create node of unknown type");

	virtual ~iNode (void) = default;

	iNode<T>* clone (void) const
	{
		return this->clone_impl();
	}

	teq::Shape shape (void)
	{
		return get_tensor()->shape();
	}

	std::string to_string (void) const
	{
		return get_tensor()->to_string();
	}

	virtual T* data (void) = 0;

	virtual void update (void) = 0;

	virtual teq::TensptrT get_tensor (void) const = 0;

protected:
	virtual iNode<T>* clone_impl (void) const = 0;
};

template <typename T>
using NodeptrT = std::shared_ptr<iNode<T>>;

template <typename T>
using NodesT = std::vector<NodeptrT<T>>;

template <typename T>
using NodeBuilderF = std::function<NodeptrT<T>(teq::TensptrT)>;

template <typename T>
struct NodeConverters final
{
	static std::unordered_map<size_t,NodeBuilderF<T>> builders_;

	static NodeptrT<T> to_node (teq::TensptrT tens)
	{
		const std::type_info& tp = typeid(*tens);
		return estd::must_getf(builders_, tp.hash_code(),
			"unknown tensor type `%s` with `%s` dtype",
			tp.name(), egen::name_type(egen::get_type<T>()).c_str())(tens);
	}

	NodeConverters (void) = delete;
};

template <typename T>
std::unordered_map<size_t,NodeBuilderF<T>> NodeConverters<T>::builders_;

template <typename TensType, typename T>
bool register_builder (NodeBuilderF<T> builder)
{
	const std::type_info& tp = typeid(TensType);
	return NodeConverters<T>::builders_.
		emplace(tp.hash_code(), builder).second;
}

#define TO_NODE(tens) NodeConverters<T>::to_node(tens)

}

#endif // ETEQ_INODE_HPP
