
#include "eteq/etens.hpp"
#include "eteq/variable.hpp"

#ifndef ETEQ_EVARS_HPP
#define ETEQ_EVARS_HPP

namespace eteq
{

template <typename T>
struct EVariable final : public ETensor<T>
{
	EVariable (void) = default;

	EVariable (VarptrT<T> vars, global::CfgMapptrT ctx = global::context()) :
		ETensor<T>(vars, ctx) {}

	friend bool operator == (const EVariable<T>& l, const EVariable<T>& r)
	{
		return l.get() == r.get();
	}

	friend bool operator != (const EVariable<T>& l, const EVariable<T>& r)
	{
		return l.get() != r.get();
	}

	operator VarptrT<T>() const
	{
		return std::static_pointer_cast<Variable<T>>(teq::TensptrT(*this));
	}

	Variable<T>* operator-> () const
	{
		return static_cast<Variable<T>*>(this->get());
	}
};

template <typename T>
using EVariablesT = std::vector<EVariable<T>>;

}

#endif // ETEQ_EVARS_HPP
