
#ifndef DISABLE_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "testutil/common.hpp"


struct FUNCTOR : public simple::TestModel {};


static ade::Tensorptr mock_back = ade::Tensor::get(ade::Shape());


struct MockOpcode : public ade::iOpcode
{
	std::string to_string (void) const override
	{
		return "MOCK";
	}

	size_t opnum (void) const override
	{
		return 0;
	}

	ade::Tensorptr gradient (ade::ArgsT args, size_t gradidx) const override
	{
		return mock_back;
	}

	ade::Tensorptr grad_vertical_merge (ade::MappedTensor bot, ade::MappedTensor top) const override
	{
		return top.tensor_;
	}

	ade::Tensorptr grad_horizontal_merge (ade::ArgsT& grads) const override
	{
		return grads.front().tensor_;
	}
};


TEST_F(FUNCTOR, Childrens)
{
	simple::SessionT sess = get_session("FUNCTOR::Childrens");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	ade::Tensorptr func = ade::Functor::get(std::make_shared<MockOpcode>(),
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, func.get());

	ade::ArgsT refs = static_cast<ade::iFunctor*>(func.get())->get_children();

	ASSERT_EQ(2, refs.size());
	EXPECT_EQ(leaf.get(), refs[0].tensor_.get());
	EXPECT_EQ(leaf1.get(), refs[1].tensor_.get());
}


TEST_F(FUNCTOR, ToString)
{
	simple::SessionT sess = get_session("FUNCTOR::ToString");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	ade::Tensorptr func = ade::Functor::get(
		std::move(std::make_unique<MockOpcode>()),
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, func.get());

	EXPECT_STREQ("MOCK", func->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
