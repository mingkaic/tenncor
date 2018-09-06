#include <memory>

#include "ade/shape.hpp"

#include "util/error.hpp"

#ifndef ADE_TENSOR_HPP
#define ADE_TENSOR_HPP

namespace ade
{

struct Tensorptr;

struct iTensor
{
	iTensor (Shape& shape) :
		shape_(shape) {}

	virtual ~iTensor (void) {}

	virtual Tensorptr gradient (Tensorptr& leaf) const = 0;

	virtual std::string to_string (void) const = 0;

	Shape shape_;
};

struct Tensorptr
{
	Tensorptr (iTensor& tens) :
		ptr_(&tens) {}

	Tensorptr (iTensor* tens) :
		ptr_(tens)
	{
		if (nullptr == tens)
		{
			util::handle_error("init nodeptr with nullptr");
		}
	}

	virtual ~Tensorptr (void) = default;

	iTensor* operator -> (void)
	{
		return ptr_.get();
	}

	const iTensor* operator -> (void) const
	{
		return ptr_.get();
	}

	iTensor* get (void) const
	{
		return ptr_.get();
	}

	std::weak_ptr<iTensor> ref (void) const
	{
		return ptr_;
	}

protected:
	std::shared_ptr<iTensor> ptr_;
};

Tensorptr constant_one (std::vector<DimT> shape);

Tensorptr constant_zero (std::vector<DimT> shape);

struct Tensor final : public iTensor
{
	static Tensorptr SYMBOLIC_ONE;
	static Tensorptr SYMBOLIC_ZERO;

	static Tensorptr get (Shape shape)
	{
		return new Tensor(shape);
	}

	Tensorptr gradient (Tensorptr& wrt) const override
	{
		std::vector<DimT> shape = wrt->shape_.as_list();
		if (this == wrt.get())
		{
			return constant_one(shape);
		}
		return constant_zero(shape);
	}

	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

private:
	Tensor (Shape shape) : iTensor(shape) {}
};

}

#endif /* ADE_TENSOR_HPP */
