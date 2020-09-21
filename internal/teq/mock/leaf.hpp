
#ifndef TEQ_MOCK_LEAF_HPP
#define TEQ_MOCK_LEAF_HPP

#include "internal/teq/teq.hpp"

#include "internal/teq/mock/device.hpp"
#include "internal/teq/mock/meta.hpp"

struct MockLeaf : public teq::iLeaf
{
	MockLeaf (void) :
		ref_(std::make_shared<MockDeviceRef>()), usage_(teq::IMMUTABLE) {}

	MockLeaf (teq::Shape shape, std::string label = "", bool cst = true) :
		ref_(std::make_shared<MockDeviceRef>()),
		shape_(shape), label_(label),
		usage_(cst ? teq::IMMUTABLE : teq::VARUSAGE) {}

	template <typename T>
	MockLeaf (const std::vector<T>& data, teq::Shape shape,
		std::string label = "", bool cst = true) :
		ref_(std::make_shared<MockDeviceRef>(data)),
		shape_(shape), label_(label),
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
		return *ref_;
	}

	const teq::iDeviceRef& device (void) const override
	{
		return *ref_;
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

	std::shared_ptr<teq::iDeviceRef> ref_ = std::make_shared<MockDeviceRef>();

	teq::Shape shape_;

	std::string label_;

	teq::Usage usage_ = teq::UNKNOWN_USAGE;

	MockMeta meta_;
};

#endif // TEQ_MOCK_LEAF_HPP
