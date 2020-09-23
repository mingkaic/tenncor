
#ifndef DISABLE_OXSVC_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "dbg/print/teq.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/serial/oxsvc/oxsvc.hpp"


const std::string testdir = "models/test";


const std::string test_service = "tenncor.serial.oxsvc.test";


struct LOAD : public ::testing::Test, public DistrTestcase
{
	LOAD (void) : DistrTestcase(test_service) {}

protected:
	void TearDown (void) override
	{
		clean_up();
	}

	distr::iDistrMgrptrT make_mgr (size_t port, const std::string& id = "")
	{
		return DistrTestcase::make_mgr(port, {
			distr::register_iosvc,
			distr::register_oxsvc,
		}, id);
	}

	void check_clean (void)
	{
		ppconsul::catalog::Catalog catalog(*consul_);
		auto services = catalog.service(service_name_);
		ASSERT_EQ(services.size(), 0);
	}
};


TEST_F(LOAD, AllLocalGraph)
{
	distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));

	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/local_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(ids, model.graph());
	EXPECT_EQ(2, graph_roots.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/local_oxsvc.txt");
	ASSERT_TRUE(expectstr.is_open());
	while (std::getline(expectstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			expect += line + '\n';
		}
	}

	PrettyEquation artist;
	artist.cfg_.showshape_ = true;
	std::stringstream gotstr;

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");
	auto root1 = ids.right.at("root1");
	auto root2 = ids.right.at("root2");
	ASSERT_NE(nullptr, root1);
	ASSERT_NE(nullptr, root2);
	artist.print(gotstr, root1);
	artist.print(gotstr, root2);

	while (std::getline(gotstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			got += line + '\n';
		}
	}

	EXPECT_STREQ(expect.c_str(), got.c_str());
}


TEST_F(LOAD, RemoteGraph)
{
	distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));
	distr::iDistrMgrptrT manager2(make_mgr(5113, "mgr2"));

	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/remote_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	distr::ox::TopographyT topography = {
		{"root1", "mgr"},
		{"root2", "mgr"},
		{"predictable_0", "mgr2"},
		{"predictable_1", "mgr2"},
		{"predictable_2", "mgr2"},
		{"predictable_3", "mgr2"},
	};
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(
		ids, model.graph(), topography);
	EXPECT_EQ(2, graph_roots.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/remote_oxsvc.txt");
	ASSERT_TRUE(expectstr.is_open());
	while (std::getline(expectstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			expect += line + '\n';
		}
	}

	PrettyEquation artist;
	artist.cfg_.showshape_ = true;
	std::stringstream gotstr;

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");
	auto root1 = ids.right.at("root1");
	auto root2 = ids.right.at("root2");
	ASSERT_NE(nullptr, root1);
	ASSERT_NE(nullptr, root2);
	artist.print(gotstr, root1);
	artist.print(gotstr, root2);

	while (std::getline(gotstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			got += line + '\n';
		}
	}

	EXPECT_STREQ(expect.c_str(), got.c_str());
}


#endif // DISABLE_OXSVC_LOAD_TEST
