#include "dbg/stream/ade.hpp"

#ifndef TEST_TUTIL_HPP
#define TEST_TUTIL_HPP

namespace tutil
{

std::string compare_graph (std::istream& expectstr, ade::TensptrT root,
	bool showshape = true, LabelsMapT labels = {});

#define EXPECT_GRAPHEQ(MSG, ROOT) {\
	std::istringstream ss(MSG);\
	auto compare_str = tutil::compare_graph(ss, ROOT);\
	EXPECT_EQ(0, compare_str.size()) << compare_str;\
}

}

#endif // TEST_TUTIL_HPP
