
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
	MockDeviceRef devref;
	MockMeta mockmeta;
	MockMeta badmeta;
	auto leaf = make_var(argshape);
	auto a = make_obs(data.data(), devref, "", 0, teq::TensptrsT{leaf});
	auto b = make_obs(data.data(), devref, "", 0, teq::TensptrsT{leaf});
	auto c = make_var(argshape);
	EXPECT_CALL(*a, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*b, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(badmeta));
	EXPECT_CALL(*b, get_meta()).WillRepeatedly(ReturnRef(badmeta));
	EXPECT_CALL(*c, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(badmeta, type_code()).WillRepeatedly(Return(0));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));

#ifdef PERM_OP
	EXPECT_CALL(Const(devref), odata()).WillRepeatedly(Invoke(
	[&data]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(data.data());
		return out;
	}));
#endif

	marsh::Maps attrs;

	std::string fatalmsg = "cannot perform `ADD` without arguments";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(eteq::Functor<double>::get(egen::ADD, teq::TensptrsT{}, std::move(attrs)), fatalmsg.c_str());

	std::string fatalmsg1 = "children types are not all the same";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(eteq::Functor<double>::get(egen::ADD, teq::TensptrsT{a, c}, std::move(attrs)), fatalmsg1.c_str());

	eteq::Functor<double>* f = eteq::Functor<double>::get(egen::ADD, {a, b}, std::move(attrs));
	teq::TensptrT ftens(f);
	auto g = eteq::Functor<double>::get(egen::SIN, {ftens}, std::move(attrs));
	teq::TensptrT gtens(g);

	auto fshape = f->shape();
	EXPECT_ARREQ(argshape, fshape);
	EXPECT_EQ(egen::ADD, f->get_opcode().code_);
	EXPECT_STREQ("ADD", f->get_opcode().name_.c_str());
	EXPECT_STREQ("ADD", f->to_string().c_str());
	EXPECT_FALSE(f->has_data());

	EXPECT_CALL(*a, has_data()).Times(2).WillRepeatedly(Return(false));
	EXPECT_CALL(*b, has_data()).Times(1).WillRepeatedly(Return(true));
	EXPECT_CALL(*a, must_initialize()).Times(1);
	std::string fatalmsg2 = "failed to initialize";
	EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg2, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg2)));
	EXPECT_FATAL(f->must_initialize(), fatalmsg2.c_str());

	EXPECT_CALL(*a, has_data()).Times(2).
		WillOnce(Return(false)).
		WillOnce(Return(true));
	EXPECT_CALL(*b, has_data()).Times(2).WillRepeatedly(Return(true));
	EXPECT_CALL(*a, must_initialize()).Times(1);
	g->must_initialize();
	EXPECT_TRUE(g->has_data());
	EXPECT_TRUE(f->has_data());

	f->uninitialize();
	EXPECT_FALSE(f->has_data());
	EXPECT_FALSE(g->has_data());

	eteq::Functor<double>* fcpy = f->clone();
	teq::TensptrT fcpytens(fcpy);

#ifdef SKIP_INIT
	EXPECT_FALSE(fcpy->has_data());
#else
	EXPECT_TRUE(fcpy->has_data());
#endif
}


TEST_F(FUNCTOR, UpdateChild)
{
	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillOnce(Return(true));

	std::vector<double> data{
		1, 2, 3, 4, 5, 6,
		1, 2, 3, 4, 5, 6};
	teq::Shape argshape({4, 3});
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto leaf = make_var(argshape);
	auto a = make_obs(data.data(), devref, "ABC", 0, teq::TensptrsT{leaf});
	auto b = make_obs(data.data(), devref, "DEF", 1, teq::TensptrsT{leaf});
	auto c = make_obs(data.data(), devref, "GHI", 2, teq::TensptrsT{leaf});
	EXPECT_CALL(*a, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*b, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*c, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*b, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*c, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(0));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("no_type"));

	MockMeta mockmeta2;
	auto leaf2 = make_var(teq::Shape({3, 4}));
	auto d = make_obs(data.data(), devref, "ZXY", 3, teq::TensptrsT{leaf2});
	auto e = make_var(argshape);
	EXPECT_CALL(*d, shape()).WillRepeatedly(Return(teq::Shape({3, 4})));
	EXPECT_CALL(*e, get_meta()).WillRepeatedly(ReturnRef(mockmeta2));
	EXPECT_CALL(mockmeta2, type_label()).WillRepeatedly(Return("DOUBLE"));

#ifdef PERM_OP
	EXPECT_CALL(Const(devref), odata()).WillRepeatedly(Invoke(
	[&data]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(data.data());
		return out;
	}));
#endif

	marsh::Maps attrs;
	auto f = eteq::Functor<double>::get(egen::ADD, {a, b}, std::move(attrs));
	teq::TensptrT ftens(f);

	EXPECT_FALSE(f->has_data());
	auto children = f->get_args();
	ASSERT_EQ(2, children.size());
	EXPECT_EQ(a, children.front());
	EXPECT_EQ(b, children.back());

	EXPECT_CALL(*a, has_data()).Times(2).WillRepeatedly(Return(true));
	EXPECT_CALL(*b, has_data()).Times(2).WillRepeatedly(Return(true));
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
		"with incompatible shape [3\\4] (requires shape [4\\3])";
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
	MockDeviceRef devref;
	MockMeta mockmeta;
	MockMeta mockmeta2;
	auto leaf = make_var(argshape);
	auto a = make_obs<double>(data.data(), devref, "ABC", 0, teq::TensptrsT{leaf});
	auto b = make_obs("DEF", 1, teq::TensptrsT{leaf});
	EXPECT_CALL(*b, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*b), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*a, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*b, shape()).WillRepeatedly(Return(argshape));
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*b, get_meta()).WillRepeatedly(ReturnRef(mockmeta2));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(0));
	EXPECT_CALL(mockmeta2, type_code()).WillRepeatedly(Return(0));

#ifdef PERM_OP
	EXPECT_CALL(Const(devref), odata()).WillRepeatedly(Invoke(
	[&data]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(data.data());
		return out;
	}));
#endif

	marsh::Maps attrs;
	auto f = eteq::Functor<double>::get(egen::ADD, {a, b}, std::move(attrs));
	teq::TensptrT ftens(f);

	ASSERT_FALSE(f->has_data());
	[this](const eteq::Functor<double>* f)
	{
		std::string fatalmsg = "cannot get device of uninitialized functor";
		EXPECT_CALL(*logger_, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
		EXPECT_FATAL(f->device(), fatalmsg.c_str());
	}(f);

	EXPECT_CALL(*a, has_data()).Times(2).WillRepeatedly(Return(true));
	EXPECT_CALL(*b, has_data()).Times(2).WillRepeatedly(Return(true));
	f->device();
	EXPECT_TRUE(f->has_data());

	EXPECT_EQ(0, f->get_meta().state_version());
	EXPECT_CALL(mockmeta, state_version()).WillRepeatedly(Return(1));
	EXPECT_CALL(mockmeta2, state_version()).WillRepeatedly(Return(1));
	EXPECT_TRUE(f->prop_version(3));
	EXPECT_EQ(1, f->get_meta().state_version());
	EXPECT_FALSE(f->prop_version(3));

	auto g = eteq::Functor<double>::get(egen::RAND_UNIF, {a, b}, std::move(attrs));
	teq::TensptrT gtens(g);
	EXPECT_TRUE(g->prop_version(3));
	EXPECT_EQ(1, g->get_meta().state_version());
	EXPECT_TRUE(g->prop_version(3));
	EXPECT_EQ(2, g->get_meta().state_version());
	EXPECT_TRUE(g->prop_version(3));
	EXPECT_EQ(3, g->get_meta().state_version());
	EXPECT_FALSE(g->prop_version(3));
}


#ifndef PERM_OP
TEST_F(FUNCTOR, Cache)
{
	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

	std::vector<double> data{
		1, 2, 3, 4, 5, 6,
		1, 2, 3, 4, 5, 6};
	teq::Shape argshape({4, 3});
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto a = make_var(data.data(), devref, argshape);
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(0));

	std::vector<double> outdata(12);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		marsh::Maps attrs;
		auto f = eteq::Functor<double>::get(egen::SIN, {a}, std::move(attrs));
		teq::TensptrT ftens(f);

		ASSERT_FALSE(f->has_data());
		f->cache_init();
		EXPECT_TRUE(f->has_data());

		auto dev = dynamic_cast<eigen::iEigen*>(&f->device());
		ASSERT_NE(nullptr, dev);

		auto outbytes = 12 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);

		eigen::RTMemptrT mem = memory;
		dev->assign(1, mem);
		dev->odata();
		EXPECT_NE(nullptr, dev->data());
	}
}
#endif


#endif // DISABLE_ETEQ_FUNCTOR_TEST
