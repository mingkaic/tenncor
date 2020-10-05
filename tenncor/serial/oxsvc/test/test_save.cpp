
#ifndef DISABLE_OXSVC_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/global/mock/mock.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/serial/oxsvc/oxsvc.hpp"


const std::string testdir = "models/test";


const std::string test_service = "tenncor.serial.oxsvc.test";


struct SAVE : public ::testing::Test, public DistrTestcase
{
protected:
	distr::iDistrMgrptrT make_mgr (size_t port, const std::string& id = "")
	{
		return DistrTestcase::make_mgr(port, {
			distr::register_iosvc,
			distr::register_oxsvc,
		}, id);
	}
};


TEST_F(SAVE, AllLocalGraph)
{
	distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));
	global::set_generator(std::make_shared<MockGenerator>());

	std::string expect_pbfile = testdir + "/local_oxsvc.onnx";
	std::string got_pbfile = "got_local_oxsvc.onnx";

	{
		onnx::ModelProto model;
		teq::TensptrsT roots;
		onnx::TensIdT ids;

		// subtree one
		teq::Shape shape({3, 7});

		teq::TensptrT osrc(eteq::make_variable<double>(
			std::vector<double>(shape.n_elems()).data(), shape, "osrc"));
		teq::TensptrT osrc2(eteq::make_variable<double>(
			std::vector<double>(shape.n_elems()).data(), shape, "osrc2"));

		{
			teq::TensptrT src(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "src"));
			teq::TensptrT src2(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "src2"));

			teq::TensptrT dest = eteq::make_functor(egen::SUB, {
				src2, eteq::make_functor(egen::POW, {
					eteq::make_functor(egen::DIV, {
						eteq::make_functor(egen::NEG, {osrc}),
						eteq::make_functor(egen::ADD, {
							eteq::make_functor(egen::SIN, {src}), src,
						}),
					}),
					osrc2,
				}),
			});
			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		// subtree two
		{
			teq::TensptrT src(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src"));
			teq::TensptrT src2(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src2"));
			teq::TensptrT src3(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src3"));

			teq::TensptrT dest = eteq::make_functor(egen::SUB, {
				src, eteq::make_functor(egen::MUL, {
					eteq::make_functor(egen::ABS, {src}),
					eteq::make_functor(egen::EXP, {src2}),
					eteq::make_functor(egen::NEG, {src3}),
				}),
			});
			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		auto topography = distr::get_oxsvc(*manager).save_graph(
			*model.mutable_graph(), roots, ids);
		EXPECT_EQ(2, topography.size());

		ASSERT_HAS(topography, "root1");
		ASSERT_HAS(topography, "root2");

		auto id = manager->get_id();
		EXPECT_STREQ(id.c_str(), topography.at("root1").c_str());
		EXPECT_STREQ(id.c_str(), topography.at("root2").c_str());

		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
}


TEST_F(SAVE, RemoteGraph)
{
	std::string expect_pbfile = testdir + "/remote_oxsvc.onnx";
	std::string got_pbfile = "got_remote_oxsvc.onnx";
	global::set_generator(std::make_shared<MockGenerator>());

	{
		distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));
		auto& svc = distr::get_iosvc(*manager);

		distr::iDistrMgrptrT manager2(make_mgr(5113, "mgr2"));
		auto& svc2 = distr::get_iosvc(*manager2);

		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;
		error::ErrptrT err = nullptr;

		// subtree one
		teq::Shape shape({3, 7});

		teq::TensptrT osrc(eteq::make_variable<double>(
			std::vector<double>(shape.n_elems()).data(), shape, "osrc"));
		teq::TensptrT osrc2(eteq::make_variable<double>(
			std::vector<double>(shape.n_elems()).data(), shape, "osrc2"));

		types::StringsT exposed_ids;
		{
			teq::TensptrT src(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "src"));
			teq::TensptrT src2(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "src2"));

			teq::TensptrT f = eteq::make_functor(egen::DIV, {
				eteq::make_functor(egen::NEG, {osrc}),
				eteq::make_functor(egen::ADD, {
					eteq::make_functor(egen::SIN, {src}), src,
				}),
			});
			std::string fid = svc.expose_node(f);

			teq::TensptrT ref = svc2.lookup_node(err, fid);
			ASSERT_NOERR(err);
			ASSERT_NE(nullptr, ref);

			teq::TensptrT dest = eteq::make_functor(egen::SUB, {
				src2, eteq::make_functor(egen::POW, {ref, osrc2}),
			});
			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
			exposed_ids.push_back(fid);
		}

		// subtree two
		{
			teq::TensptrT src(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src"));
			teq::TensptrT src2(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src2"));
			teq::TensptrT src3(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src3"));

			teq::TensptrT f = eteq::make_functor(egen::ABS, {src});

			std::string fid = svc.expose_node(f);
			std::string sid = svc.expose_node(src);
			std::string s3id = svc.expose_node(src3);

			teq::TensptrT f1_ref = svc2.lookup_node(err, fid);
			ASSERT_NOERR(err);
			ASSERT_NE(nullptr, f1_ref);
			teq::TensptrT src_ref = svc2.lookup_node(err, sid);
			ASSERT_NOERR(err);
			ASSERT_NE(nullptr, src_ref);
			teq::TensptrT src3_ref = svc2.lookup_node(err, s3id);
			ASSERT_NOERR(err);
			ASSERT_NE(nullptr, src3_ref);

			teq::TensptrT dest = eteq::make_functor(egen::SUB, {
				src_ref, eteq::make_functor(egen::MUL, {
					f1_ref,
					eteq::make_functor(egen::EXP, {src2}),
					eteq::make_functor(egen::NEG, {src3_ref}),
				}),
			});
			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
			exposed_ids.push_back(fid);
			exposed_ids.push_back(sid);
			exposed_ids.push_back(s3id);
		}

		auto topography = distr::get_oxsvc(*manager2).save_graph(*model.mutable_graph(), roots, ids);
		EXPECT_EQ(2 + exposed_ids.size(), topography.size());

		ASSERT_HAS(topography, "root1");
		ASSERT_HAS(topography, "root2");
		for (auto id : exposed_ids)
		{
			ASSERT_HAS(topography, id);
		}

		auto manager_id = manager->get_id();
		auto manager2_id = manager2->get_id();
		EXPECT_STREQ(manager2_id.c_str(), topography.at("root1").c_str());
		EXPECT_STREQ(manager2_id.c_str(), topography.at("root2").c_str());
		for (auto id : exposed_ids)
		{
			EXPECT_STREQ(manager_id.c_str(), topography.at(id).c_str());
		}

		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
}


#endif // DISABLE_OXSVC_SAVE_TEST
