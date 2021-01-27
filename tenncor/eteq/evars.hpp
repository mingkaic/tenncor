
#ifndef ETEQ_EVARS_HPP
#define ETEQ_EVARS_HPP

#include "tenncor/eteq/etens.hpp"
#include "tenncor/eteq/variable.hpp"

namespace eteq
{

struct EVariable final : public ETensor
{
	EVariable (void) = default;

	EVariable (VarptrT vars, const global::CfgMapptrT& ctx = global::context()) :
		ETensor(vars, ctx) {}

	friend bool operator == (const EVariable& l, const EVariable& r)
	{
		return l.get() == r.get();
	}

	friend bool operator != (const EVariable& l, const EVariable& r)
	{
		return l.get() != r.get();
	}

	operator VarptrT() const
	{
		return std::static_pointer_cast<Variable>(teq::TensptrT(*this));
	}

	Variable* operator-> () const
	{
		return static_cast<Variable*>(this->get());
	}
};

using EVariablesT = std::vector<EVariable>;

}

#endif // ETEQ_EVARS_HPP
