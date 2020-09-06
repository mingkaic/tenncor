#include "internal/teq/objs.hpp"

#ifdef TEQ_OBJS_HPP

namespace teq
{

TensorObj::TensorObj (TensptrT tens) : tens_(tens) {}

TensorObj* TensorObj::clone (void) const
{
	return static_cast<TensorObj*>(clone_impl());
}

TensorRef* TensorObj::copynreplace (TensptrT tens) const
{
	return new TensorObj(tens);
}

size_t TensorObj::class_code (void) const
{
	static const std::type_info& tp = typeid(TensorObj);
	return tp.hash_code();
}

std::string TensorObj::to_string (void) const
{
	return tens_->to_string();
}

bool TensorObj::equals (const marsh::iObject& other) const
{
	if (other.class_code() != this->class_code())
	{
		return false;
	}
	return tens_ == static_cast<const TensorObj*>(&other)->tens_;
}

void TensorObj::accept (marsh::iMarshaler& marshaler) const
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

TensptrT& TensorObj::get_tensor (void)
{
	return tens_;
}

const TensptrT& TensorObj::get_tensor (void) const
{
	return tens_;
}

marsh::iObject* TensorObj::clone_impl (void) const
{
	return new TensorObj(*this);
}

LayerObj::LayerObj (const std::string& opname, TensptrT input) :
	opname_(opname), input_(input)
{
	if (nullptr == input)
	{
		global::fatalf("cannot `%s` with null input", opname.c_str());
	}
}

LayerObj* LayerObj::clone (void) const
{
	return static_cast<LayerObj*>(clone_impl());
}

TensorRef* LayerObj::copynreplace (TensptrT tens) const
{
	return new LayerObj(opname_, tens);
}

size_t LayerObj::class_code (void) const
{
	static const std::type_info& tp = typeid(LayerObj);
	return tp.hash_code();
}

std::string LayerObj::to_string (void) const
{
	return opname_;
}

bool LayerObj::equals (const marsh::iObject& other) const
{
	if (other.class_code() != this->class_code())
	{
		return false;
	}
	auto olayer = static_cast<const LayerObj*>(&other);
	return opname_ == olayer->opname_ && input_ ==  olayer->input_;
}

void LayerObj::accept (marsh::iMarshaler& marshaler) const
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

TensptrT& LayerObj::get_tensor (void)
{
	return input_;
}

const TensptrT& LayerObj::get_tensor (void) const
{
	return input_;
}

std::string LayerObj::get_opname (void) const
{
	return opname_;
}

marsh::iObject* LayerObj::clone_impl (void) const
{
	return new LayerObj(*this);
}

iTeqMarshaler::~iTeqMarshaler (void) = default;

}

#endif
