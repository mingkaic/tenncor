#include "estd/estd.hpp"

#include "ade/itensor.hpp"

#include "ead/eigen.hpp"

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

template <typename T>
using NodeBuilderF = std::function<NodeptrT<T>(ade::TensptrT)>;

template <typename T>
struct NodeConverters final
{
	static std::unordered_map<size_t,NodeBuilderF<T>> builders_;

	static NodeptrT<T> to_node (ade::TensptrT tens)
	{
		const std::type_info& tp = typeid(*tens);
		return estd::must_getf(builders_, tp.hash_code(),
			"unknown tensor type `%s` with `%s` dtype",
			tp.name(), age::name_type(age::get_type<T>()).c_str())(tens);
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

}

#endif // EAD_INODE_HPP
