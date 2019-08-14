
#include <thread>

#include "gtest/gtest.h"

#include "perf/measure.hpp"

int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


static void mock_measure (perf::PerfRecord& record)
{
	perf::MeasureScope _defer(&record, "f1");

	std::this_thread::sleep_for(
		std::chrono::milliseconds(1000));
}


TEST(PERFORMANCE, Measure)
{
	perf::PerfRecord record;
	mock_measure(record);

	std::stringstream ss;
	record.to_csv(ss);
	auto got = ss.str();
	EXPECT_STREQ("f1,", got.substr(0, 3).c_str());
	EXPECT_EQ(8, got.size()); // expect f1,xxxx\n
}
