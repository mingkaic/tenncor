
#ifndef DISABLE_SESSION_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/test/common.hpp"

#include "teq/session.hpp"


struct MockOpfunc final : public teq::iOperableFunc
{
	MockOpfunc (teq::TensptrT a) :
		args_({teq::identity_map(a)}) {}

	MockOpfunc (teq::TensptrT a, teq::TensptrT b) :
		args_({teq::identity_map(a), teq::identity_map(b)}),
		shape_(args_[0].shape()) {}

	/// Implementation of iTensor
	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return "";
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return teq::Opcode{};
	}

	/// Implementation of iFunctor
	const teq::ArgsT& get_children (void) const override
	{
		return args_;
	}

	/// Implementation of iFunctor
	void update_child (teq::FuncArg arg, size_t index) override {}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		updated_ = true;
	}

	/// Implementation of iData
	void* data (void) override
	{
		return nullptr;
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		return nullptr;
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return 0;
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return "";
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return 0;
	}

	bool updated_ = false;

	teq::ArgsT args_;

	teq::Shape shape_;
};


TEST(SESSION, Track)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));
	teq::TensptrT d(new MockTensor(shape));

	teq::TensptrT x(new MockOpfunc(a, b));
	teq::TensptrT target(new MockOpfunc(x, c));
	teq::TensptrT target2(new MockOpfunc(x, d));

	// this tests if session can track be called multiple times
	teq::Session session;
	session.track({target});

	// expect session.ops_ to contain x and target
	EXPECT_ARRHAS(session.ops_, x.get());
	EXPECT_ARRHAS(session.ops_, target.get());
	EXPECT_EQ(2, session.ops_.size());

	// expect target to be tracked
	EXPECT_HAS(session.tracked_, target);
	EXPECT_EQ(1, session.tracked_.size());

	session.track({target2});

	// expect session.ops_ to contain all the ops
	EXPECT_ARRHAS(session.ops_, x.get());
	EXPECT_ARRHAS(session.ops_, target.get());
	EXPECT_ARRHAS(session.ops_, target2.get());
	EXPECT_EQ(3, session.ops_.size());

	// expect both targets to be tracked
	EXPECT_HAS(session.tracked_, target);
	EXPECT_HAS(session.tracked_, target2);
	EXPECT_EQ(2, session.tracked_.size());
}


TEST(SESSION, Update)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(a, b);
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(x, c);

	// * (target) = not updated
	// `-- + (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(x->updated_);

	teq::Session session;
	session.track({target});
	session.update();

	// expected state:
	// * (target) = updated
	// `-- + (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c

	EXPECT_TRUE(target->updated_);
	EXPECT_TRUE(x->updated_);
}


TEST(SESSION, UpdateIgnore)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));
	teq::TensptrT d(new MockTensor(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(a, b);
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(x, c);
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(y, d);

	// - (target) = not updated
	// `-- * (y) = not updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(y->updated_);
	ASSERT_FALSE(x->updated_);

	teq::Session session;
	session.track({target});
	session.update({y.get()});

	// expected state:
	// - (target) = updated
	// `-- * (y) = not updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	EXPECT_TRUE(target->updated_);
	EXPECT_FALSE(x->updated_);
	EXPECT_FALSE(y->updated_);

	target->updated_ = x->updated_ = y->updated_ = false;

	session.update({x.get()});

	// expected state:
	// - (target) = updated
	// `-- * (y) = updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	EXPECT_TRUE(target->updated_);
	EXPECT_TRUE(y->updated_);
	EXPECT_FALSE(x->updated_);
}


TEST(SESSION, UpdateIgnoreCommonDesc)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));

	std::shared_ptr<MockOpfunc> u = std::make_shared<MockOpfunc>(a);
	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(u, b);
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(c, u);
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(y, x);

	// - (target) = not updated
	// `-- / (y) = not updated
	// |   `-- c
	// |   `-- - (u) = not updated
	// |       `-- a
	// `-- * (x) = not updated
	//     `-- - (u)
	//     |   `-- a
	//     `-- b

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(y->updated_);
	ASSERT_FALSE(x->updated_);
	ASSERT_FALSE(u->updated_);

	teq::Session session;
	session.track({target});
	session.update({y.get()});

	// expected state:
	// - (target) = updated
	// `-- / (y) = not updated
	// |   `-- c
	// |   `-- - (u) = updated
	// |       `-- a
	// `-- * (x) = updated
	//     `-- - (u)
	//     |   `-- a
	//     `-- b

	EXPECT_TRUE(target->updated_);
	EXPECT_FALSE(y->updated_);
	EXPECT_TRUE(x->updated_);
	EXPECT_TRUE(u->updated_);
}


TEST(SESSION, TargetedUpdate)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(a, b);
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(x, c);

	// * (target) = not updated
	// `-- + (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(x->updated_);

	teq::Session session;
	session.track({target});
	session.update_target(teq::TensSetT{x.get()});

	// expected state:
	// * (target) = not updated
	// `-- + (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c

	EXPECT_FALSE(target->updated_);
	EXPECT_TRUE(x->updated_);

}


TEST(SESSION, TargetedUpdateIgnore)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));
	teq::TensptrT d(new MockTensor(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(a, b);
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(x, c);
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(y, d);

	// - (targetd) = not updated
	// `-- * (y) = not updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(y->updated_);
	ASSERT_FALSE(x->updated_);

	teq::Session session;
	session.track({target});
	session.update_target({y.get()}, {x.get()});

	// expected state:
	// - (targetd) = not updated
	// `-- * (y) = updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	EXPECT_FALSE(target->updated_);
	EXPECT_TRUE(y->updated_);
	EXPECT_FALSE(x->updated_);
}


TEST(SESSION, TargetedUpdateIgnoreCommonDesc)
{
	teq::Shape shape;

	teq::TensptrT a(new MockTensor(shape));
	teq::TensptrT b(new MockTensor(shape));
	teq::TensptrT c(new MockTensor(shape));
	teq::TensptrT d(new MockTensor(shape));

	std::shared_ptr<MockOpfunc> u = std::make_shared<MockOpfunc>(a);
	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(u, b);
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(c, u);
	std::shared_ptr<MockOpfunc> z = std::make_shared<MockOpfunc>(y, x);
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(z, d);

	// pow (targeted) = not updated
	// `-- - (z) = not updated
	// |   `-- / (y) = not updated
	// |   |   `-- c
	// |   |   `-- - (u) = not updated
	// |   |       `-- a
	// |   `-- * (x) = not updated
	// |       `-- - (u)
	// |       |   `-- a
	// |       `-- b
	// `-- d

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(z->updated_);
	ASSERT_FALSE(y->updated_);
	ASSERT_FALSE(x->updated_);
	ASSERT_FALSE(u->updated_);

	teq::Session session;
	session.track({target});
	session.update_target({z.get()}, {y.get()});

	// expected state:
	// pow (targeted) = not updated
	// `-- - (z) = updated
	// |   `-- / (y) = not updated
	// |   |   `-- c
	// |   |   `-- - (u) = updated
	// |   |       `-- a
	// |   `-- * (x) = updated
	// |       `-- - (u)
	// |       |   `-- a
	// |       `-- b
	// `-- d

	EXPECT_FALSE(target->updated_);
	EXPECT_TRUE(z->updated_);
	EXPECT_FALSE(y->updated_);
	EXPECT_TRUE(x->updated_);
	EXPECT_TRUE(u->updated_);
}


#endif // DISABLE_SESSION_TEST
