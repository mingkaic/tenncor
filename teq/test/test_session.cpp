
#ifndef DISABLE_SESSION_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"
#include "teq/mock/opfunc.hpp"

#include "teq/session.hpp"


TEST(SESSION, Track)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	teq::TensptrT x(new MockOpfunc(teq::TensptrsT{a, b},
		std::vector<double>{}));
	teq::TensptrT target(new MockOpfunc(teq::TensptrsT{x, c},
		std::vector<double>{}));
	teq::TensptrT target2(new MockOpfunc(teq::TensptrsT{x, d},
		std::vector<double>{}));

	// this tests if session can track be called multiple times
	teq::Session session;
	session.track({target});
	EXPECT_EQ(1, session.get_tracked().size());

	// expect session.ops_ to contain x and target
	EXPECT_ARRHAS(session.ops_, x.get());
	EXPECT_ARRHAS(session.ops_, target.get());
	EXPECT_EQ(2, session.ops_.size());

	// expect target to be tracked
	EXPECT_HAS(session.tracked_, target);
	EXPECT_EQ(1, session.tracked_.size());

	session.track({target2});
	EXPECT_EQ(2, session.get_tracked().size());

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

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{a, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{x, c},
		std::vector<double>{});

	// * (target) = not updated
	// `-- + (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(x->updated_);

	teq::Session session;
	session.track({target});
	EXPECT_EQ(1, session.get_tracked().size());
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

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{a, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(teq::TensptrsT{x, c},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{y, d},
		std::vector<double>{});

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
	EXPECT_EQ(1, session.get_tracked().size());
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

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> u = std::make_shared<MockOpfunc>(teq::TensptrsT{a},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{u, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(teq::TensptrsT{c, u},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{y, x},
		std::vector<double>{});

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
	EXPECT_EQ(1, session.get_tracked().size());
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

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{a, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{x, c},
		std::vector<double>{});

	// * (target) = not updated
	// `-- + (x) = not updated
	// |   `-- a
	// |   `-- b
	// `-- c

	ASSERT_FALSE(target->updated_);
	ASSERT_FALSE(x->updated_);

	teq::Session session;
	session.track({target});
	EXPECT_EQ(1, session.get_tracked().size());
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

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{a, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(teq::TensptrsT{x, c},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{y, d},
		std::vector<double>{});

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
	EXPECT_EQ(1, session.get_tracked().size());
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

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> u = std::make_shared<MockOpfunc>(teq::TensptrsT{a},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{u, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(teq::TensptrsT{c, u},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> z = std::make_shared<MockOpfunc>(teq::TensptrsT{y, x},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{z, d},
		std::vector<double>{});

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
	EXPECT_EQ(1, session.get_tracked().size());
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


TEST(SESSION, Clear)
{
	teq::Shape shape;

	teq::TensptrT a(new MockLeaf(shape));
	teq::TensptrT b(new MockLeaf(shape));
	teq::TensptrT c(new MockLeaf(shape));
	teq::TensptrT d(new MockLeaf(shape));

	std::shared_ptr<MockOpfunc> u = std::make_shared<MockOpfunc>(teq::TensptrsT{a},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> x = std::make_shared<MockOpfunc>(teq::TensptrsT{u, b},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> y = std::make_shared<MockOpfunc>(teq::TensptrsT{c, u},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> z = std::make_shared<MockOpfunc>(teq::TensptrsT{y, x},
		std::vector<double>{});
	std::shared_ptr<MockOpfunc> target = std::make_shared<MockOpfunc>(teq::TensptrsT{z, d},
		std::vector<double>{});

	teq::Session session;
	session.track({target});
	EXPECT_EQ(1, session.get_tracked().size());
	session.clear();
	EXPECT_EQ(0, session.get_tracked().size());
	session.update();

	EXPECT_FALSE(target->updated_);
	EXPECT_FALSE(z->updated_);
	EXPECT_FALSE(y->updated_);
	EXPECT_FALSE(x->updated_);
	EXPECT_FALSE(u->updated_);
}


TEST(SESSION, FailTrack)
{
	teq::Shape shape;
	teq::TensptrT a(new MockLeaf(shape));
	std::shared_ptr<MockFunctor> b = std::make_shared<MockFunctor>(
		teq::TensptrsT{a}, teq::Opcode{"bad_func", 0});

	teq::Session session;
	EXPECT_FATAL(session.track({b}),
		"cannot track non-operable functor bad_func");
}


#endif // DISABLE_SESSION_TEST
