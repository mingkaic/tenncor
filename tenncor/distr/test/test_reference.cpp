
#ifndef DISABLE_DISTR_REFERENCE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/distr.hpp"


TEST(REFERENCE, Meta)
{
	std::string cluster_id = "test_local";
	std::string remote_str = "abcdef";
	teq::Shape shape({3, 2});

	auto a = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id, "a", remote_str);
	auto acpy = a->clone();

	auto& ameta = a->get_meta();

	EXPECT_EQ(egen::DOUBLE, ameta.type_code());
	EXPECT_STREQ("DOUBLE", ameta.type_label().c_str());
	EXPECT_EQ(sizeof(double), ameta.type_size());

	EXPECT_EQ(0, ameta.state_version());

	EXPECT_EQ(teq::PLACEHOLDER, a->get_usage());

	EXPECT_STREQ("test_local/a", a->to_string().c_str());
	EXPECT_STREQ("test_local/a", acpy->to_string().c_str());

	EXPECT_STREQ(remote_str.c_str(), a->remote_string().c_str());
	EXPECT_STREQ(remote_str.c_str(), acpy->remote_string().c_str());

	delete acpy;
}


TEST(REFERENCE, Data)
{
	std::string cluster_id = "test_local";
	teq::Shape shape({3, 2});

	auto a = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id, "a", "abcdef");

	[](const distr::DistrRef& ref)
	{
		auto ptr = (const double*) ref.device().data();
		EXPECT_EQ(0, *ptr);
	}(*a);

	auto ptr = (const double*) a->device().data();
	auto& ameta = a->get_meta();

	std::vector<double> initdata = {1, 2, 3, 4, 5, 6};
	std::vector<double> data2 = {2, 9, 7, 2, 1, 5};

	a->update_data(initdata.data(), 1);
	EXPECT_EQ(1, ameta.state_version());
	std::vector<double> avec(ptr, ptr + 6);
	EXPECT_ARREQ(initdata, avec);

	a->update_data(data2.data(), 1);
	EXPECT_EQ(1, ameta.state_version());
	std::vector<double> avec2(ptr, ptr + 6);
	EXPECT_ARREQ(initdata, avec2);

	a->update_data(data2.data(), 2);
	EXPECT_EQ(2, ameta.state_version());
	std::vector<double> avec3(ptr, ptr + 6);
	EXPECT_ARREQ(data2, avec3);
}


TEST(REFERENCE, Reachability)
{
	std::string cluster_id = "test_local";
	teq::Shape shape({3, 2});

	auto a = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id, "a", "abcdef");
	auto b = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id, "b", "bcdefg");
	auto c = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id, "c", "cdefgh");

	teq::TensptrT d(new MockFunctor(teq::TensptrsT{c}, teq::Opcode{"MOCK2", 0}));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{a, b}, teq::Opcode{"MOCK1", 1}));
	teq::TensptrT g(new MockFunctor(teq::TensptrsT{d, f}, teq::Opcode{"MOCK0", 0}));

	auto everything = distr::reachable_refs(teq::TensptrsT{g});
	auto ab = distr::reachable_refs(teq::TensptrsT{g}, teq::TensSetT{d.get()});
	auto bc = distr::reachable_refs(teq::TensptrsT{g}, teq::TensSetT{a.get()});

	EXPECT_EQ(3, everything.size());
	EXPECT_EQ(2, ab.size());
	EXPECT_EQ(2, bc.size());
	EXPECT_HAS(everything, a.get());
	EXPECT_HAS(everything, b.get());
	EXPECT_HAS(everything, c.get());
	EXPECT_HAS(ab, a.get());
	EXPECT_HAS(ab, b.get());
	EXPECT_HAS(bc, b.get());
	EXPECT_HAS(bc, c.get());
}


TEST(REFERENCE, Seperate)
{
	std::string cluster_id = "test_local";
	std::string cluster_id2 = "test_local2";
	std::string cluster_id3 = "test_local3";
	std::string cluster_id4 = "test_local4";
	teq::Shape shape({3, 2});

	auto a = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id3, "a", "abcdef");
	auto b = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id, "b", "bcdefg");
	auto c = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id2, "c", "cdefgh");
	auto d = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id4, "d", "defghi");
	auto e = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id2, "e", "efghij");
	auto f = std::make_shared<distr::DistrRef>(
		egen::DOUBLE, shape, cluster_id3, "f", "fghijk");

	types::StrUMapT<types::StrUSetT> servers;
	distr::separate_by_server(servers, distr::DRefSetT{
		a.get(), b.get(), c.get(), d.get(), e.get(), f.get()});
	EXPECT_EQ(4, servers.size());
	EXPECT_HAS(servers, cluster_id);
	EXPECT_HAS(servers, cluster_id2);
	EXPECT_HAS(servers, cluster_id3);
	EXPECT_HAS(servers, cluster_id4);

	EXPECT_EQ(1, servers[cluster_id].size());
	EXPECT_EQ(2, servers[cluster_id2].size());
	EXPECT_EQ(2, servers[cluster_id3].size());
	EXPECT_EQ(1, servers[cluster_id4].size());
	EXPECT_HAS(servers[cluster_id], "b");
	EXPECT_HAS(servers[cluster_id2], "c");
	EXPECT_HAS(servers[cluster_id2], "e");
	EXPECT_HAS(servers[cluster_id3], "a");
	EXPECT_HAS(servers[cluster_id3], "f");
	EXPECT_HAS(servers[cluster_id4], "d");
}


#endif // DISABLE_DISTR_REFERENCE_TEST
