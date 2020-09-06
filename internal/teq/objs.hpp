
#ifndef TEQ_OBJS_HPP
#define TEQ_OBJS_HPP

#include "internal/marsh/marsh.hpp"

#include "internal/teq/itensor.hpp"

namespace teq
{

const std::string layer_attr = "layer";

struct TensorRef : public marsh::iObject
{
	virtual TensptrT& get_tensor (void) = 0;

	virtual const TensptrT& get_tensor (void) const = 0;

	virtual TensorRef* copynreplace (TensptrT) const = 0;
};

struct TensorObj final : public TensorRef
{
	TensorObj (TensptrT tens);

	TensorObj* clone (void) const;

	TensorRef* copynreplace (TensptrT tens) const override;

	size_t class_code (void) const override;

	std::string to_string (void) const override;

	bool equals (const marsh::iObject& other) const override;

	void accept (marsh::iMarshaler& marshaler) const override;

	TensptrT& get_tensor (void) override;

	const TensptrT& get_tensor (void) const override;

private:
	marsh::iObject* clone_impl (void) const override;

	TensptrT tens_;
};

struct LayerObj final : public TensorRef
{
	LayerObj (const std::string& opname, TensptrT input);

	LayerObj* clone (void) const;

	TensorRef* copynreplace (TensptrT tens) const override;

	size_t class_code (void) const override;

	std::string to_string (void) const override;

	bool equals (const marsh::iObject& other) const override;

	void accept (marsh::iMarshaler& marshaler) const override;

	TensptrT& get_tensor (void) override;

	const TensptrT& get_tensor (void) const override;

	std::string get_opname (void) const;

private:
	marsh::iObject* clone_impl (void) const override;

	std::string opname_;

	TensptrT input_;
};

struct iTeqMarshaler : public marsh::iMarshaler
{
	virtual ~iTeqMarshaler (void);

	virtual void marshal (const marsh::String& str) = 0;

	virtual void marshal (const marsh::iNumber& num) = 0;

	virtual void marshal (const marsh::iArray& arr) = 0;

	virtual void marshal (const marsh::iTuple& tup) = 0;

	virtual void marshal (const marsh::Maps& mm) = 0;

	virtual void marshal (const TensorObj& tens) = 0;

	virtual void marshal (const LayerObj& layer) = 0;
};

}

#endif // TEQ_OBJS_HPP
