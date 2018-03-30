#include "gtest/gtest.h"

#include "utils/utils.hpp"

#include "fuzz/irng.hpp"

#include "proto/serial/data.pb.h"

int main(int argc, char **argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	testify::set_generator(nnutils::get_generator);
	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	// delete all global objects allocated by libprotobuf
	google::protobuf::ShutdownProtobufLibrary();
	return ret;
}
