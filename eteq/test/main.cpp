
#include "gtest/gtest.h"

#include "onnx/onnx.pb.h"

#include "exam/exam.hpp"

#include "teq/logs.hpp"

#include "eigen/device.hpp"
#include "eigen/random.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	LOG_INIT(exam::TestLogger);
	DEVICE_INIT(eigen::Device);
	RANDOM_INIT;

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
