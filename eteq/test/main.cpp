
#include "gtest/gtest.h"

#include "onnx/onnx.pb.h"

#include "exam/exam.hpp"

int main (int argc, char** argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	set_logger(std::static_pointer_cast<logs::iLogger>(exam::tlogger));

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
