
#ifndef DISABLE_DISTRIB_TEST

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "experimental/distrib/session.hpp"

#include "eteq/make.hpp"
#include "eteq/derive.hpp"

#include "generated/api.hpp"


TEST(DISTRIB, Init)
{
	eigen::Device device;
	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	std::vector<double> data3 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	// cluster 1
	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> src2 = eteq::make_constant<double>(data2.data(), shape);
	eteq::ETensor<double> dest = src + src2;

	distrib::DSessptrT sess = std::make_shared<distrib::DistribSess>("127.0.0.1:5112");
	sess->track(teq::TensptrSetT{dest});
	std::string id = sess->lookup_id(dest);

	// cluster 2
	eteq::ETensor<double> src3 = eteq::make_constant<double>(data3.data(), shape);
	auto dest2 = eteq::ETensor<double>(sess->lookup_node(id)) * src3;

	distrib::DSessptrT sess2 = std::make_shared<distrib::DistribSess>("127.0.0.1:5113");
	sess2->track(teq::TensptrSetT{dest2});
}


#endif // DISABLE_DISTRIB_TEST
