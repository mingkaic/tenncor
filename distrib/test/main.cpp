
#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "query/query.pb.h"

#include "teq/logs.hpp"

#include "eigen/eigen.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	LOG_INIT(exam::TestLogger);
	RANDOM_INIT;

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
