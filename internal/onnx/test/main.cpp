#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/global/global.hpp"

#include "internal/onnx/onnx.pb.h"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	global::set_logger(new exam::TestLogger());

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
