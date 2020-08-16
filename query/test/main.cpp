
#include "gtest/gtest.h"

#include "query/query.pb.h"

#include "exam/exam.hpp"

#include "global/global.hpp"

#include "eigen/eigen.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	global::set_logger(new exam::TestLogger());

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
