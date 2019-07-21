
#ifndef DISABLE_GRAD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ade/test/common.hpp"

#include "ade/grad_def.hpp"


struct MockGradientBuilder final : public ade::iGradientBuilder
{
    ade::TensptrT local_derivative (ade::FuncptrT op, size_t arg_idx) const override
    {
        return nullptr;
    }

    ade::TensptrT chain_rule (ade::FuncptrT op, const ade::TensptrT& local_der,
		ade::TensptrT supcomp_grad, size_t arg_idx) const override
    {
        return nullptr;
    }

    ade::TensptrT get_const_one (ade::Shape shape) const override
    {
        return nullptr;
    }

    ade::TensptrT get_const_zero (ade::Shape shape) const override
    {
        return nullptr;
    }

	ade::TensptrT add (ade::TensptrT& lhs, ade::TensptrT& rhs) const override
    {
        return nullptr;
    }
};


TEST(GRAD, Builder)
{
    MockGradientBuilder builder;
}


#endif // DISABLE_GRAD_TEST
