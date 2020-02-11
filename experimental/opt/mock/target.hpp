
#include "experimental/opt/target.hpp"

#ifndef OPT_MOCK_TARGET_HPP
#define OPT_MOCK_TARGET_HPP

struct MockTarget final : public opt::iTarget
{
    MockTarget (teq::TensptrT tag) : tag_(tag) {}

    teq::TensptrT convert (const query::SymbMapT& candidates) const override
    {
        return tag_;
    }

    teq::TensptrT tag_;
};

#endif // OPT_MOCK_TARGET_HPP
