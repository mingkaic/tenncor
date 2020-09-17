
#ifndef DISABLE_ETEQ_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"

#include "internal/eigen/mock/mock.hpp"


TEST(FUNCTOR, Initiation)
{
	teq::Shape argshape({4, 3});
	std::vector<double> data{
		1, 2, 3, 4, 5, 6,
		1, 2, 3, 4, 5, 6};
	auto a = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{});
	auto b = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{});
	auto c = std::make_shared<MockLeaf>(argshape);
	c->meta_.tcode_ = egen::DOUBLE;

	marsh::Maps attrs;

	EXPECT_FATAL(eteq::Functor<double>::get(
		egen::ADD, teq::TensptrsT{}, std::move(attrs)),
		"cannot perform `ADD` without arguments");

	EXPECT_FATAL(eteq::Functor<double>::get(
		egen::ADD, teq::TensptrsT{a, c}, std::move(attrs)),
		"children types are not all the same");

	eteq::Functor<double>* f = eteq::Functor<double>::get(
		egen::ADD, {a, b}, std::move(attrs));
	teq::TensptrT ftens(f);
	auto g = eteq::Functor<double>::get(
		egen::SIN, {ftens}, std::move(attrs));
	teq::TensptrT gtens(g);

	auto fshape = f->shape();
	EXPECT_ARREQ(argshape, fshape);
	EXPECT_EQ(egen::ADD, f->get_opcode().code_);
	EXPECT_STREQ("ADD", f->get_opcode().name_.c_str());
	EXPECT_STREQ("ADD", f->to_string().c_str());
	EXPECT_FALSE(f->has_data());

	a->succeed_initial_ = false;
	EXPECT_FATAL(f->must_initialize(), "failed to initialize");
	a->succeed_initial_ = true;

	g->must_initialize();
	EXPECT_TRUE(g->has_data());
	EXPECT_TRUE(f->has_data());

	f->uninitialize();
	EXPECT_FALSE(f->has_data());
	EXPECT_FALSE(g->has_data());

	eteq::Functor<double>* fcpy = f->clone();
	teq::TensptrT fcpytens(fcpy);

	EXPECT_TRUE(fcpy->has_data());
}


TEST(FUNCTOR, UpdateChild)
{
	std::vector<double> data{
		1, 2, 3, 4, 5, 6,
		1, 2, 3, 4, 5, 6};
	teq::Shape argshape({4, 3});
	auto a = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{"ABC", 0});
	auto b = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{"DEF", 1});
	auto c = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{"GHI", 2});

	auto d = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(teq::Shape({3, 4}))},
		data, teq::Opcode{"ZXY", 3});
	auto e = std::make_shared<MockLeaf>(argshape);
	e->meta_.tname_ = "DOUBLE";

	marsh::Maps attrs;
	auto f = eteq::Functor<double>::get(
		egen::ADD, {a, b}, std::move(attrs));
	teq::TensptrT ftens(f);

	EXPECT_FALSE(f->has_data());
	auto children = f->get_args();
	ASSERT_EQ(2, children.size());
	EXPECT_EQ(a, children.front());
	EXPECT_EQ(b, children.back());

	f->must_initialize();

	f->update_child(c, 1);
	EXPECT_FALSE(f->has_data());

	children = f->get_args();
	ASSERT_EQ(2, children.size());
	EXPECT_EQ(a, children.front());
	EXPECT_EQ(c, children.back());

	f->update_child(c, 0);
	EXPECT_FALSE(f->has_data());

	children = f->get_args();
	ASSERT_EQ(2, children.size());
	EXPECT_EQ(c, children.front());
	EXPECT_EQ(c, children.back());

	EXPECT_FATAL(f->update_child(d, 1), "cannot update child 1 to argument "
		"with incompatible shape [3\\4\\1\\1\\1\\1\\1\\1] (requires shape "
		"[4\\3\\1\\1\\1\\1\\1\\1])");

	EXPECT_FATAL(f->update_child(e, 0), "cannot update child 0 to argument "
		"with different type DOUBLE (requires type no_type)");

	EXPECT_FATAL(f->update_child(a, 2),
		"cannot replace argument 2 when only there are only 2 available");

	delete f;
}


TEST(FUNCTOR, Prop)
{
	std::vector<double> data{
		1, 2, 3, 4, 5, 6,
		1, 2, 3, 4, 5, 6};
	teq::Shape argshape({4, 3});
	auto a = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{"ABC", 0});
	auto b = std::make_shared<MockObservable>(teq::TensptrsT{
		std::make_shared<MockLeaf>(argshape)},
		data, teq::Opcode{"DEF", 1});

	marsh::Maps attrs;
	auto f = eteq::Functor<double>::get(
		egen::ADD, {a, b}, std::move(attrs));
	teq::TensptrT ftens(f);

	ASSERT_FALSE(f->has_data());
	[](const eteq::Functor<double>* f)
	{
		EXPECT_FATAL(f->device(), "cannot get device of uninitialized functor");
	}(f);

	f->device();
	EXPECT_TRUE(f->has_data());

	EXPECT_EQ(0, f->get_meta().state_version());
	a->func_.meta_.version_ = 1;
	b->func_.meta_.version_ = 1;
	EXPECT_TRUE(f->prop_version(3));
	EXPECT_EQ(1, f->get_meta().state_version());
	EXPECT_FALSE(f->prop_version(3));

	auto g = eteq::Functor<double>::get(
		egen::RAND_UNIF, {a, b}, std::move(attrs));
	teq::TensptrT gtens(g);
	EXPECT_TRUE(g->prop_version(3));
	EXPECT_EQ(1, g->get_meta().state_version());
	EXPECT_TRUE(g->prop_version(3));
	EXPECT_EQ(2, g->get_meta().state_version());
	EXPECT_TRUE(g->prop_version(3));
	EXPECT_EQ(3, g->get_meta().state_version());
	EXPECT_FALSE(g->prop_version(3));
}


#endif // DISABLE_ETEQ_FUNCTOR_TEST
