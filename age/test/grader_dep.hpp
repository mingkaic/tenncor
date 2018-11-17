#include "ade/tensor.hpp"
#include "bwd/grader.hpp"

#ifndef MOCK_GRADER_DEP_HPP
#define MOCK_GRADER_DEP_HPP

const size_t khaled_constant = 45;

struct MockTensor : public ade::Tensor
{
    MockTensor (double scalar, ade::Shape shape) :
        scalar_(scalar), shape_(shape) {}

	/// Return the shape held by this tensor
	const ade::Shape& shape (void) const override
	{
        return shape_;
    }

	/// Return the string representation of the tensor
	std::string to_string (void) const override
	{
        return "MockTensor";
    }

	char* data (void) override
	{
        return (char*) &scalar_;
    }

	const char* data (void) const override
	{
        return (const char*) &scalar_;
    }

	size_t type_code (void) const override
	{
        return 0;
    }

    double scalar_;

    ade::Shape shape_;
};

ade::Tensorptr arms_heavy (size_t idx, age::TensT args);

ade::Tensorptr dj_grad (age::TensT args, size_t idx);

#endif // MOCK_GRADER_DEP_HPP
