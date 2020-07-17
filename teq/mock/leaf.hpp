#include "teq/ileaf.hpp"

#include "teq/mock/device.hpp"
#include "teq/mock/meta.hpp"

#ifndef TEQ_MOCK_LEAF_HPP
#define TEQ_MOCK_LEAF_HPP

struct MockLeaf : public teq::iLeaf
{
	MockLeaf (void) : usage_(teq::IMMUTABLE) {}

	MockLeaf (teq::Shape shape, std::string label = "", bool cst = true) :
		shape_(shape), label_(label),
		usage_(cst ? teq::IMMUTABLE : teq::VARUSAGE) {}

	MockLeaf (std::vector<double> data, teq::Shape shape,
		std::string label = "", bool cst = true) :
		ref_(data), shape_(shape), label_(label),
		usage_(cst ? teq::IMMUTABLE : teq::VARUSAGE) {}

	virtual ~MockLeaf (void) = default;

	teq::Shape shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return label_;
	}

	teq::iDeviceRef& device (void) override
	{
		return ref_;
	}

	const teq::iDeviceRef& device (void) const override
	{
		return ref_;
	}

	const teq::iMetadata& get_meta (void) const override
	{
		return meta_;
	}

	teq::Usage get_usage (void) const override
	{
		return usage_;
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockLeaf(*this);
	}

	MockDeviceRef ref_;

	teq::Shape shape_;

	std::string label_;

	teq::Usage usage_ = teq::UNKNOWN_USAGE;

	MockMeta meta_;
};

#endif // TEQ_MOCK_LEAF_HPP
