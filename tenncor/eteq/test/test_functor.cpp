
#ifndef DISABLE_ETEQ_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "tenncor/eteq/eteq.hpp"

#include "internal/eigen/mock/mock.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


struct FUNCTOR : public tutil::TestcaseWithLogger<> {};


TEST_F(FUNCTOR, Initiation)
{
	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

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

	std::string fatalmsg = "cannot perform `ADD` without arguments";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(eteq::Functor<double>::get(
		egen::ADD, teq::TensptrsT{}, std::move(attrs)), fatalmsg.c_str());

	std::string fatalmsg1 = "children types are not all the same";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(eteq::Functor<double>::get(
		egen::ADD, teq::TensptrsT{a, c}, std::move(attrs)), fatalmsg1.c_str());

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
	std::string fatalmsg2 = "failed to initialize";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg2, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg2)));
	EXPECT_FATAL(f->must_initialize(), fatalmsg2.c_str());
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


TEST_F(FUNCTOR, UpdateChild)
{
	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillOnce(Return(true));

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

	std::string fatalmsg = "cannot update child 1 to argument "
		"with incompatible shape [3\\4\\1\\1\\1\\1\\1\\1] (requires shape "
		"[4\\3\\1\\1\\1\\1\\1\\1])";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(f->update_child(d, 1), fatalmsg.c_str());

	std::string fatalmsg1 = "cannot update child 0 to argument "
		"with different type DOUBLE (requires type no_type)";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(f->update_child(e, 0), fatalmsg1.c_str());

	std::string fatalmsg2 = "cannot replace argument 2 when only there are only 2 available";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg2, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg2)));
	EXPECT_FATAL(f->update_child(a, 2), fatalmsg2.c_str());
}


TEST_F(FUNCTOR, Prop)
{
	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

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
	[this](const eteq::Functor<double>* f)
	{
		std::string fatalmsg = "cannot get device of uninitialized functor";
		EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
		EXPECT_FATAL(f->device(), fatalmsg.c_str());
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
