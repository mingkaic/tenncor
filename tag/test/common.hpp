#include "teq/ileaf.hpp"

#ifndef TAG_TEST_COMMON_HPP
#define TAG_TEST_COMMON_HPP

struct MockTensor final : public teq::iLeaf
{
	MockTensor (void) = default;

	MockTensor (teq::Shape shape) : shape_(shape) {}

	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

	void* data (void) override
	{
		return nullptr;
	}

	const void* data (void) const override
	{
		return nullptr;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	std::string type_label (void) const override
	{
		return "";
	}

	size_t nbytes (void) const override
	{
		return 0;
	}

	bool is_const (void) const override
	{
		return true;
	}

	teq::Shape shape_;
};

#endif // TAG_TEST_COMMON_HPP
