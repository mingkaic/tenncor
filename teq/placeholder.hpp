#include "teq/signature.hpp"

#ifndef TEQ_PLACEHOLDER_HPP
#define TEQ_PLACEHOLDER_HPP

namespace teq
{

struct Placeholder final : public iTensor, public iSignature
{
	Placeholder (ShapeSignature shape, std::string label = "") :
		shape_(shape), label_(label) {}

	/// Return deep copy of this Functor
	Placeholder* clone (void) const
	{
		return static_cast<Placeholder*>(clone_impl());
	}

	void assign (const DataptrT& input)
	{
		Shape shape = input->data_shape();
		if (false == shape.compatible_after(shape_, 0))
		{
			logs::fatalf("assigning data shaped %s to tensor %s",
				shape.to_string().c_str(), shape_.to_string().c_str());
		}
		data_ = input;
	}

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Implementation of iTensor
	Shape shape (void) const override
	{
		if (can_build())
		{
			logs::fatal("cannot get shape from unassigned placeholder");
		}
		return data_->data_shape();
	}

    /// Implementation of iTensor
    std::string to_string (void) const override
    {
        return label_;
    }

	/// Implementation of iSignature
	bool can_build (void) const override
	{
		return nullptr != data_;
	}

	/// Implementation of iSignature
	DataptrT build_data (void) const override
	{
		if (false == can_build())
		{
			logs::fatal("cannot get tensor from unassigned placeholder");
		}
		return data_;
	}

	/// Implementation of iSignature
	ShapeSignature shape_sign (void) const override
	{
		return shape_;
	}

	DataptrT get_child (void) const
	{
		return data_;
	}

private:
	Placeholder (const Placeholder& other) = default;

	iTensor* clone_impl (void) const override
	{
		return new Placeholder(*this);
	}

	ShapeSignature shape_;

	DataptrT data_ = nullptr;

	std::string label_;
};

/// Smart pointer of placeholder nodes to preserve assign functions
using PlaceptrT = std::shared_ptr<Placeholder>;

}

#endif // TEQ_PLACEHOLDER_HPP
