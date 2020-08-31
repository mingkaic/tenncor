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

struct FindTensAttr final : public marsh::iMarshaler
{
	void marshal (const marsh::String& num) override {}

	void marshal (const marsh::iNumber& num) override {}

	void marshal (const marsh::iArray& arr) override
	{
		arr.foreach([this](size_t,const marsh::iObject* obj){ process(obj); });
	}

	void marshal (const marsh::iTuple& tup) override
	{
		tup.foreach([this](size_t,const marsh::iObject* obj){ process(obj); });
	}

	void marshal (const marsh::Maps& mm) override
	{
		auto keys = mm.ls_attrs();
		for (auto key : keys)
		{
			process(mm.get_attr(key));
		}
	}

	void process (const marsh::iObject* obj)
	{
		if (nullptr == obj)
		{
			return;
		}
		if (auto dep = dynamic_cast<const teq::TensorRef*>(obj))
		{
			deps_.push_back(dep->get_tensor());
		}
		else
		{
			obj->accept(*this);
		}
	}

	teq::TensptrsT deps_;
};

/// Encoding of operation
struct Opcode final
{
	/// String representation of operation
	std::string name_;

	/// Numerical encoding of operation
	size_t code_;
};

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

	/// Return vector of functor dependencies including attribute tensor refs
	virtual teq::TensptrsT get_dependencies (void) const
	{
		auto deps = get_args();
		if (this->size() > 0)
		{
			marsh::Maps attrs;
			marsh::get_attrs(attrs, *this);
			FindTensAttr attrf;
			attrs.accept(attrf);
			auto& subdeps = attrf.deps_;
			deps.insert(deps.end(), subdeps.begin(), subdeps.end());
		}
		return deps;
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
