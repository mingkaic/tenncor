
#include "gtest/gtest.h"

#include "internal/onnx/onnx.hpp"
#include "internal/query/query.pb.h"

#include "testutil/tutil.hpp"

#include "internal/global/global.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	global::set_logger(new exam::NoSupportLogger());

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
