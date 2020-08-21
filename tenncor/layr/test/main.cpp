
#include "gtest/gtest.h"

#include "internal/onnx/onnx.pb.h"

#include "internal/global/global.hpp"

#include "internal/eigen/eigen.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
