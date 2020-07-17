
#ifndef DISABLE_DISTRIB_TEST

#include <cstdlib>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "distrib/session.hpp"

#include "eteq/make.hpp"
#include "eteq/derive.hpp"

#include "generated/api.hpp"


const std::string test_service = "tenncor_distrib_ctest";


ppconsul::Consul create_test_consul (void)
{
	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	return ppconsul::Consul{address};
}


struct DISTRIB : public ::testing::Test
{
	DISTRIB (void) : consul_(create_test_consul()) {}

protected:
	void TearDown (void) override
	{
		ppconsul::agent::Agent agent(consul_);
		ppconsul::catalog::Catalog catalog(consul_);
		auto services = catalog.service(test_service);
		for (auto& service : services)
		{
			agent.deregisterService(service.second.id);
		}
	}

	void check_clean (void)
	{
		ppconsul::catalog::Catalog catalog(consul_);
		auto services = catalog.service(test_service);
		ASSERT_EQ(services.size(), 0);
	}

	distrib::DSessptrT make_sess (size_t port, const std::string& id = "")
	{
		return std::make_shared<distrib::DistribSess>(
			consul_, port, test_service, id, distrib::ClientConfig(
				std::chrono::milliseconds(5000),
				std::chrono::milliseconds(10000),
				5
			));
	}

	ppconsul::Consul consul_;
};


TEST_F(DISTRIB, SharingNodes)
{
	{
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		// cluster 1
		distrib::DSessptrT sess = make_sess(5112, "sess1");

		eteq::ETensor<double> src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor<double> src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor<double> dest = src + src2;
		sess->track(teq::TensptrSetT{dest});
		std::string id = *sess->lookup_id(dest);

		// cluster 2
		distrib::DSessptrT sess2 = make_sess(5113, "sess2");

		eteq::ETensor<double> src3 =
			eteq::make_constant<double>(data3.data(), shape);
		err::ErrptrT err = nullptr;
		auto ref = sess2->lookup_node(err, id);
		std::string err_msg;
		if (nullptr != err)
		{
			err_msg = err->to_string();
		}
		EXPECT_STREQ("", err_msg.c_str());
		ASSERT_NE(nullptr, ref);
		auto dest2 = eteq::ETensor<double>(ref) * src3;
		sess2->track(teq::TensptrSetT{dest2});

		eteq::ETensor<double> src4 =
			eteq::make_constant<double>(data.data(), shape);
		auto bad_id = sess->lookup_id(src4);
		EXPECT_FALSE(bad_id);
	}
	check_clean();
}


TEST_F(DISTRIB, DataPassing)
{
	{
		eigen::Device device;
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		// cluster 1
		distrib::DSessptrT sess = make_sess(5112, "sess1");

		eteq::ETensor<double> src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor<double> src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor<double> dest = src + src2;
		sess->track(teq::TensptrSetT{dest});
		std::string id = *sess->lookup_id(dest);

		// cluster 2
		distrib::DSessptrT sess2 = make_sess(5113, "sess2");

		eteq::ETensor<double> src3 =
			eteq::make_constant<double>(data3.data(), shape);
		err::ErrptrT err = nullptr;
		auto ref = sess2->lookup_node(err, id);
		std::string err_msg;
		if (nullptr != err)
		{
			err_msg = err->to_string();
		}
		EXPECT_STREQ("", err_msg.c_str());
		ASSERT_NE(nullptr, ref);
		auto dest2 = eteq::ETensor<double>(ref) * src3;
		sess2->track(teq::TensptrSetT{dest2});

		ASSERT_TRUE(sess->get_dependencies().empty());
		ASSERT_EQ(sess2->get_dependencies().size(), 1);

		sess2->update_target(device, teq::TensSetT{dest2.get()});

		std::vector<double> exdata = {
			2997, 3430, 68, 10626, 4956, 8241,
			5016, 4956, 4554, 13156, 12384, 2088,
			4895, 4620, 2480, 5832, 2240, 2430,
			2100, 7140, 8908, 520, 2940, 4600,
		};
		auto gotshape = dest2->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* goptr = (double*) dest2->device().data();
		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(exdata[i], goptr[i]);
		}
	}
	check_clean();
}


#endif // DISABLE_DISTRIB_TEST
