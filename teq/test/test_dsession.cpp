
#ifndef DISABLE_DYNAMIC_SESSION_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "teq/isession.hpp"


static MockDevice mdevice;


TEST(DYNAMIC_SESSION, Track)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	teq::TensptrT x(new MockFunctor(teq::TensptrsT{a, b}));
	teq::TensptrT target(new MockFunctor(teq::TensptrsT{x, c}));
	teq::TensptrT target2(new MockFunctor(teq::TensptrsT{x, d}));

	// this tests if session can track be called multiple times
	teq::Session session;
	session.track({target});

	// expect session.ops_ to contain x and target
	ASSERT_EQ(2, session.ops_.size());
	EXPECT_EQ(1, session.ops_[0].size());
	EXPECT_EQ(1, session.ops_[1].size());
	EXPECT_ARRHAS(session.ops_[0], x.get());
	EXPECT_ARRHAS(session.ops_[1], target.get());

	session.track({target2});

	// expect session.ops_ to contain all the ops
	EXPECT_EQ(2, session.ops_.size());
	EXPECT_EQ(1, session.ops_[0].size());
	EXPECT_EQ(2, session.ops_[1].size());
	EXPECT_ARRHAS(session.ops_[0], x.get());
	EXPECT_ARRHAS(session.ops_[1], target.get());
	EXPECT_ARRHAS(session.ops_[1], target2.get());
}


TEST(DYNAMIC_SESSION, Update)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));

	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{a, b});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{x, c});

	// * (target) = not updated
	// `-- + (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	ASSERT_FALSE(target->data_.ref_.updated_);
	ASSERT_FALSE(x->data_.ref_.updated_);

	teq::Session session;
	session.track({target});
	session.update_target(mdevice, {target.get()});

	// expected state:
	// * (target) = updated
	// `-- + (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c

	EXPECT_TRUE(target->data_.ref_.updated_);
	EXPECT_TRUE(x->data_.ref_.updated_);
}


TEST(DYNAMIC_SESSION, UpdateIgnore)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{a, b}, teq::Opcode{"+", 0});
	auto y = std::make_shared<MockFunctor>(teq::TensptrsT{x, c}, teq::Opcode{"*", 1});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{y, d}, teq::Opcode{"-", 2});

	// - (target) = not updated
	// `-- * (y) = not updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	ASSERT_FALSE(target->data_.ref_.updated_);
	ASSERT_FALSE(y->data_.ref_.updated_);
	ASSERT_FALSE(x->data_.ref_.updated_);

	teq::Session session;
	session.track({target});
	session.update_target(mdevice, {target.get()}, {y.get()});

	// expected state:
	// - (target) = updated
	// `-- * (y) = not updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	EXPECT_TRUE(target->data_.ref_.updated_);
	EXPECT_FALSE(x->data_.ref_.updated_);
	EXPECT_FALSE(y->data_.ref_.updated_);

	target->data_.ref_.updated_ = x->data_.ref_.updated_ = y->data_.ref_.updated_ = false;

	session.update_target(mdevice, {target.get()}, {x.get()});

	// expected state:
	// - (target) = updated
	// `-- * (y) = updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	EXPECT_TRUE(target->data_.ref_.updated_);
	EXPECT_TRUE(y->data_.ref_.updated_);
	EXPECT_FALSE(x->data_.ref_.updated_);
}


TEST(DYNAMIC_SESSION, UpdateIgnoreCommonDesc)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));

	auto u = std::make_shared<MockFunctor>(teq::TensptrsT{a});
	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{u, b});
	auto y = std::make_shared<MockFunctor>(teq::TensptrsT{c, u});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{y, x});

	// - (target) = not updated
	// `-- / (y) = not updated
	// |   `-- c
	// |   `-- - (u) = not updated
	// |       `-- a
	// `-- * (x) = not updated
	//     `-- - (u)
	//     |   `-- a
	//     `-- b

	ASSERT_FALSE(target->data_.ref_.updated_);
	ASSERT_FALSE(y->data_.ref_.updated_);
	ASSERT_FALSE(x->data_.ref_.updated_);
	ASSERT_FALSE(u->data_.ref_.updated_);

	teq::Session session;
	session.track({target});
	session.update_target(mdevice, {target.get()}, {y.get()});

	// expected state:
	// - (target) = updated
	// `-- / (y) = not updated
	// |   `-- c
	// |   `-- - (u) = updated
	// |       `-- a
	// `-- * (x) = updated
	//     `-- - (u)
	//     |   `-- a
	//     `-- b

	EXPECT_TRUE(target->data_.ref_.updated_);
	EXPECT_FALSE(y->data_.ref_.updated_);
	EXPECT_TRUE(x->data_.ref_.updated_);
	EXPECT_TRUE(u->data_.ref_.updated_);
}


TEST(DYNAMIC_SESSION, TargetedUpdate)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));

	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{a, b});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{x, c});

	// * (target) = not updated
	// `-- + (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	ASSERT_FALSE(target->data_.ref_.updated_);
	ASSERT_FALSE(x->data_.ref_.updated_);

	teq::Session session;
	session.track({target});
	session.update_target(mdevice, teq::TensSetT{x.get()});

	// expected state:
	// * (target) = not updated
	// `-- + (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c

	EXPECT_FALSE(target->data_.ref_.updated_);
	EXPECT_TRUE(x->data_.ref_.updated_);

}


TEST(DYNAMIC_SESSION, TargetedUpdateIgnore)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{a, b});
	auto y = std::make_shared<MockFunctor>(teq::TensptrsT{x, c});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{y, d});

	// - (targetd) = not updated
	// `-- * (y) = not updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	ASSERT_FALSE(target->data_.ref_.updated_);
	ASSERT_FALSE(y->data_.ref_.updated_);
	ASSERT_FALSE(x->data_.ref_.updated_);

	teq::Session session;
	session.track({target});
	session.update_target(mdevice, {y.get()}, {x.get()});

	// expected state:
	// - (targetd) = not updated
	// `-- * (y) = updated
	// |   `-- + (x) = not updated
	// |   |   `-- a
	// |   |   `-- b
	// |   `-- c
	// `-- d

	EXPECT_FALSE(target->data_.ref_.updated_);
	EXPECT_TRUE(y->data_.ref_.updated_);
	EXPECT_FALSE(x->data_.ref_.updated_);
}


TEST(DYNAMIC_SESSION, TargetedUpdateIgnoreCommonDesc)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	auto u = std::make_shared<MockFunctor>(teq::TensptrsT{a});
	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{u, b});
	auto y = std::make_shared<MockFunctor>(teq::TensptrsT{c, u});
	auto z = std::make_shared<MockFunctor>(teq::TensptrsT{y, x});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{z, d});

	// pow (targeted) = not updated
	// `-- - (z) = not updated
	// |   `-- / (y) = not updated
	// |   |   `-- c
	// |   |   `-- - (u) = not updated
	// |   |       `-- a
	// |   `-- * (x) = not updated
	// |       `-- - (u)
	// |       |   `-- a
	// |       `-- b
	// `-- d

	ASSERT_FALSE(target->data_.ref_.updated_);
	ASSERT_FALSE(z->data_.ref_.updated_);
	ASSERT_FALSE(y->data_.ref_.updated_);
	ASSERT_FALSE(x->data_.ref_.updated_);
	ASSERT_FALSE(u->data_.ref_.updated_);

	teq::Session session;
	session.track({target});
	session.update_target(mdevice, {z.get()}, {y.get()});

	// expected state:
	// pow (targeted) = not updated
	// `-- - (z) = updated
	// |   `-- / (y) = not updated
	// |   |   `-- c
	// |   |   `-- - (u) = updated
	// |   |       `-- a
	// |   `-- * (x) = updated
	// |       `-- - (u)
	// |       |   `-- a
	// |       `-- b
	// `-- d

	EXPECT_FALSE(target->data_.ref_.updated_);
	EXPECT_TRUE(z->data_.ref_.updated_);
	EXPECT_FALSE(y->data_.ref_.updated_);
	EXPECT_TRUE(x->data_.ref_.updated_);
	EXPECT_TRUE(u->data_.ref_.updated_);
}


TEST(DYNAMIC_SESSION, Clear)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	auto u = std::make_shared<MockFunctor>(teq::TensptrsT{a});
	auto x = std::make_shared<MockFunctor>(teq::TensptrsT{u, b});
	auto y = std::make_shared<MockFunctor>(teq::TensptrsT{c, u});
	auto z = std::make_shared<MockFunctor>(teq::TensptrsT{y, x});
	auto target = std::make_shared<MockFunctor>(teq::TensptrsT{z, d});

	teq::Session session;
	session.track({target});
	ASSERT_EQ(4, session.ops_.size());
	EXPECT_EQ(1, session.ops_[0].size());
	EXPECT_EQ(2, session.ops_[1].size());
	EXPECT_EQ(1, session.ops_[2].size());
	EXPECT_EQ(1, session.ops_[3].size());
	session.clear();

	EXPECT_EQ(0, session.ops_.size());
	auto err = session.update_target(mdevice, {target.get()});
	EXPECT_ERR(err, "not all targets are tracked");
	EXPECT_FALSE(target->data_.ref_.updated_);
	EXPECT_FALSE(z->data_.ref_.updated_);
	EXPECT_FALSE(y->data_.ref_.updated_);
	EXPECT_FALSE(x->data_.ref_.updated_);
	EXPECT_FALSE(u->data_.ref_.updated_);
}


#endif // DISABLE_DYNAMIC_SESSION_TEST
