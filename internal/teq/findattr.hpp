#ifndef TEQ_FINDATTR_HPP
#define TEQ_FINDATTR_HPP

#include "internal/teq/objs.hpp"

namespace teq
{

struct FindTensAttr final : public iTeqMarshaler
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

	void marshal (const TensorObj& tens) override
	{
		process(&tens);
	}

	void marshal (const LayerObj& layer) override
	{
		process(&layer);
	}

	void process (const marsh::iObject* obj)
	{
		if (nullptr == obj)
		{
			return;
		}
		if (auto dep = dynamic_cast<const TensorRef*>(obj))
		{
			tens_.push_back(dep->get_tensor());
		}
		else
		{
			obj->accept(*this);
		}
	}

	TensptrsT tens_;
};

}

#endif // TEQ_FINDATTR_HPP
