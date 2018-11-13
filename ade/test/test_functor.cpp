
#ifndef DISABLE_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "testutil/common.hpp"


struct FUNCTOR : public simple::TestModel {};


struct MockTensor final : public ade::Tensor
{
	MockTensor (void) = default;

	MockTensor (ade::Shape shape) : shape_(shape) {}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

	char* data (void) override
	{
		return nullptr;
	}

	const char* data (void) const override
	{
		return nullptr;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	ade::Shape shape_;
};


TEST_F(FUNCTOR, Childrens)
{
	simple::SessionT sess = get_session("FUNCTOR::Childrens");

	ade::Tensorptr leaf = new MockTensor();
	ade::Tensorptr leaf1 = new MockTensor();

	ade::Tensorptr func = ade::Functor::get(ade::Opcode{"MOCK", 0},
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

	ade::Tensorptr leaf = new MockTensor();
	ade::Tensorptr leaf1 = new MockTensor();

	ade::Tensorptr func = ade::Functor::get(ade::Opcode{"MOCK", 0},
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, func.get());

	EXPECT_STREQ("MOCK", func->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
