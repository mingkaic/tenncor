///
/// functor.hpp
/// teq
///
/// Purpose:
/// Define functor nodes of an equation graph
///

#ifndef TEQ_IFUNCTOR_HPP
#define TEQ_IFUNCTOR_HPP

#include "internal/teq/objs.hpp"

namespace teq
{

const std::string dependency_key = "dependencies";

const std::string layer_key = "layer";

/// Encoding of operation
struct Opcode final
{
	/// String representation of operation
	std::string name_;

	/// Numerical encoding of operation
	size_t code_;
};

using TensptrRefT = std::reference_wrapper<TensptrT>;

using TensptrCRefT = std::reference_wrapper<const TensptrT>;

using TensptrRefsT = std::vector<TensptrRefT>;

using TensptrCRefsT = std::vector<TensptrCRefT>;

/// Interface of iOperation-defined operation node
struct iFunctor : public iTensor, public marsh::iAttributed
{
	virtual ~iFunctor (void) = default;

	iFunctor* clone (void) const
	{
		return static_cast<iFunctor*>(this->clone_impl());
	}

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Return operation encoding
	virtual Opcode get_opcode (void) const = 0;

	/// Return vector of functor arguments
	virtual TensptrsT get_args (void) const = 0;

	/// Return vector of functor dependencies
	TensptrRefsT get_dependencies (void)
	{
		TensptrRefsT deps;
		if (auto tensattr = dynamic_cast<TensArrayT*>(
			get_attr(dependency_key)))
		{
			deps.reserve(tensattr->size());
			tensattr->foreach(
			[&](size_t, marsh::iObject* obj)
			{
				deps.push_back(
					static_cast<TensorObj*>(obj)->get_tensor());
			});
		}
		return deps;
	}

	TensptrCRefsT get_dependencies (void) const
	{
		TensptrCRefsT deps;
		if (auto tensattr = dynamic_cast<const TensArrayT*>(
			get_attr(dependency_key)))
		{
			deps.reserve(tensattr->size());
			tensattr->foreach(
			[&](size_t, const marsh::iObject* obj)
			{
				deps.push_back(
					static_cast<const TensorObj*>(obj)->get_tensor());
			});
		}
		return deps;
	}

	void add_dependencies (TensptrsT dependencies)
	{
		auto deps_attr = dynamic_cast<TensArrayT*>(
			get_attr(dependency_key));
		if (nullptr == deps_attr)
		{
			add_attr(dependency_key,
				std::make_unique<TensArrayT>());
			deps_attr = static_cast<TensArrayT*>(
				get_attr(dependency_key));
		}
		auto& contents = deps_attr->contents_;
		for (auto& tens : dependencies)
		{
			contents.emplace(contents.end(),
				std::make_unique<TensorObj>(tens));
		}
	}

	/// Return vector of functor arguments and dependencies attributes
	TensptrsT get_argndeps (void) const
	{
		auto argndeps = get_args();
		auto deps = get_dependencies();
		argndeps.insert(argndeps.end(), deps.begin(), deps.end());
		return argndeps;
	}

	/// Update child at specified index
	virtual void update_child (TensptrT arg, size_t index) = 0;
};

using FuncptrT = std::shared_ptr<iFunctor>;

using FuncsT = std::vector<iFunctor*>;

using FuncSetT = std::unordered_set<iFunctor*>;

template <typename T>
using FuncMapT = std::unordered_map<iFunctor*,T>;

}

#endif // TEQ_IFUNCTOR_HPP
