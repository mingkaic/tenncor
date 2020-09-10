
#ifndef DISABLE_TEQ_EVALUATOR_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"

#include "internal/teq/evaluator.hpp"


static MockDevice mdevice;


TEST(EVALUATOR, SetGet)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();

	auto& eval = teq::get_eval(ctx);
	void* origptr = &eval;
	auto oeval = new teq::Evaluator();
	teq::set_eval(oeval, ctx);
	EXPECT_NE(origptr, &teq::get_eval(ctx));
	EXPECT_EQ(oeval, &teq::get_eval(ctx));
}


TEST(EVALUATOR, Update)
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

	teq::Evaluator eval;
	eval.evaluate(mdevice, {target.get()});

	// expected state:
	// * (target) = updated
	// `-- + (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c

	EXPECT_TRUE(target->data_.ref_.updated_);
	EXPECT_TRUE(x->data_.ref_.updated_);
}


TEST(EVALUATOR, UpdateIgnore)
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

	teq::Evaluator eval;
	eval.evaluate(mdevice, {target.get()}, {y.get()});

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

	eval.evaluate(mdevice, {target.get()}, {x.get()});

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


TEST(EVALUATOR, UpdateIgnoreCommonDesc)
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

	teq::Evaluator eval;
	eval.evaluate(mdevice, {target.get()}, {y.get()});

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


TEST(EVALUATOR, TargetedUpdate)
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

	teq::Evaluator eval;
	eval.evaluate(mdevice, teq::TensSetT{x.get()});

	// expected state:
	// * (target) = not updated
	// `-- + (x) = updated
	// |   `-- a
	// |   `-- b
	// `-- c

	EXPECT_FALSE(target->data_.ref_.updated_);
	EXPECT_TRUE(x->data_.ref_.updated_);

}


TEST(EVALUATOR, TargetedUpdateIgnore)
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

	teq::Evaluator eval;
	eval.evaluate(mdevice, {y.get()}, {x.get()});

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


TEST(EVALUATOR, TargetedUpdateIgnoreCommonDesc)
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

	teq::Evaluator eval;
	eval.evaluate(mdevice, {z.get()}, {y.get()});

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


#endif // DISABLE_TEQ_EVALUATOR_TEST
