#include <fstream>

#include <google/protobuf/util/json_util.h>

#include "flag/flag.hpp"
#include "fmts/fmts.hpp"

#include "experimental/cce/weights.pb.h"

int main (int argc, const char** argv)
{
	std::string readpath;
	std::string writepath;
	flag::FlagSet flags("inspector");
	flags.add_flags()
		("read", flag::opt::value<std::string>(&readpath),
			"filename of model to inspect")
		("write", flag::opt::value<std::string>(&writepath),
			"filename to write json format");

	if (false == flags.parse(argc, argv))
	{
		return 1;
	}

	logs::get_logger().set_log_level(logs::INFO);

	std::ifstream readstr(readpath);
	if (readstr.is_open())
	{
		weights::OpWeights ops;
		if (false == ops.ParseFromIstream(&readstr))
		{
			logs::fatalf("failed to parse from istream when read file %s",
				readpath.c_str());
		}
		std::string jsonstr;
		google::protobuf::util::JsonPrintOptions options;
		options.add_whitespace = true;
		if (google::protobuf::util::Status::OK !=
			google::protobuf::util::MessageToJsonString(
				ops, &jsonstr, options))
		{
			logs::fatal("failed to parse op weights");
		}

		std::ofstream writestr(writepath);
		if (writestr.is_open())
		{
			writestr << jsonstr;
			writestr.flush();
			writestr.close();
		}
		else
		{
			std::cout << jsonstr << std::endl;
		}

		readstr.close();
	}
	else
	{
		logs::warnf("failed to read file `%s`", readpath.c_str());
	}

	return 0;
}
