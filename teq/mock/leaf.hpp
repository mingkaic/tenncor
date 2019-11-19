#include "teq/ileaf.hpp"

#ifndef TEQ_MOCK_LEAF_HPP
#define TEQ_MOCK_LEAF_HPP

struct MockTensor : public teq::iLeaf
{
	MockTensor (std::string label = "") : label_(label) {}

	MockTensor (teq::Shape shape,
		std::string label = "") :
		shape_(shape), label_(label) {}

	MockTensor (teq::Shape shape,
		std::string label, bool cst) :
		shape_(shape), label_(label), cst_(cst) {}

	virtual ~MockTensor (void) = default;

	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return label_;
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
		return cst_;
	}

	teq::Shape shape_;

	std::string label_;

	bool cst_ = true;
};

#endif // TEQ_MOCK_LEAF_HPP
