
#ifndef TEQ_OBJS_HPP
#define TEQ_OBJS_HPP

#include "internal/marsh/marsh.hpp"

#include "internal/teq/itensor.hpp"

namespace teq
{

const std::string layer_attr = "layer";

struct TensorObj;

struct LayerObj;

struct iTeqMarshaler : public marsh::iMarshaler
{
	virtual ~iTeqMarshaler (void) = default;

	virtual void marshal (const marsh::String& str) = 0;

	virtual void marshal (const marsh::iNumber& num) = 0;

	virtual void marshal (const marsh::iArray& arr) = 0;

	virtual void marshal (const marsh::iTuple& tup) = 0;

	virtual void marshal (const marsh::Maps& mm) = 0;

	virtual void marshal (const TensorObj& tens) = 0;

	virtual void marshal (const LayerObj& layer) = 0;
};

struct TensorRef : public marsh::iObject
{
	virtual TensptrT& get_tensor (void) = 0;

	virtual const TensptrT& get_tensor (void) const = 0;

	virtual TensorRef* copynreplace (TensptrT) const = 0;
};

struct TensorObj final : public TensorRef
{
	TensorObj (TensptrT tens) : tens_(tens) {}

	TensorObj* clone (void) const
	{
		return static_cast<TensorObj*>(clone_impl());
	}

	TensorRef* copynreplace (TensptrT tens) const override
	{
		return new TensorObj(tens);
	}

	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(TensorObj);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		return tens_->to_string();
	}

	bool equals (const marsh::iObject& other) const override
	{
		if (other.class_code() != this->class_code())
		{
			return false;
		}
		return tens_ == static_cast<const TensorObj*>(&other)->tens_;
	}

	void accept (marsh::iMarshaler& marshaler) const override
	{
		if (auto marsh = dynamic_cast<iTeqMarshaler*>(&marshaler))
		{
			marsh->marshal(*this);
		}
		else
		{
			global::warn("non-teq marshaler cannot marshal "
				"tensor-typed objects");
		}
	}

	TensptrT& get_tensor (void) override
	{
		return tens_;
	}

	const TensptrT& get_tensor (void) const override
	{
		return tens_;
	}

private:
	marsh::iObject* clone_impl (void) const override
	{
		return new TensorObj(*this);
	}

	TensptrT tens_;
};

struct LayerObj final : public TensorRef
{
	LayerObj (const std::string& opname, TensptrT input) :
		opname_(opname), input_(input)
	{
		if (nullptr == input)
		{
			global::fatalf("cannot `%s` with null input", opname.c_str());
		}
	}

	LayerObj* clone (void) const
	{
		return static_cast<LayerObj*>(clone_impl());
	}

	TensorRef* copynreplace (TensptrT tens) const override
	{
		return new LayerObj(opname_, tens);
	}

	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(LayerObj);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		return opname_;
	}

	bool equals (const marsh::iObject& other) const override
	{
		if (other.class_code() != this->class_code())
		{
			return false;
		}
		auto olayer = static_cast<const LayerObj*>(&other);
		return opname_ == olayer->opname_ && input_ ==  olayer->input_;
	}

	void accept (marsh::iMarshaler& marshaler) const override
	{
		if (auto marsh = dynamic_cast<iTeqMarshaler*>(&marshaler))
		{
			marsh->marshal(*this);
		}
		else
		{
			global::warn("non-teq marshaler cannot marshal "
				"layer-typed objects");
		}
	}

	TensptrT& get_tensor (void) override
	{
		return input_;
	}

	const TensptrT& get_tensor (void) const override
	{
		return input_;
	}

	std::string get_opname (void) const
	{
		return opname_;
	}

private:
	marsh::iObject* clone_impl (void) const override
	{
		return new LayerObj(*this);
	}

	std::string opname_;

	TensptrT input_;
};

using TensArrayT = marsh::PtrArray<TensorObj>;

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

#endif // TEQ_OBJS_HPP
