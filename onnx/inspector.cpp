#include <fstream>

#include <google/protobuf/util/json_util.h>

#include "flag/flag.hpp"
#include "fmts/fmts.hpp"
#include "estd/strs.hpp"

#include "onnx/onnx.pb.h"

const std::string onnx_ext = ".onnx";

void write_json (std::string writepath, const std::string& jsonstr)
{
	if (estd::has_affix(writepath, onnx_ext))
	{
		std::fstream writestr(writepath, std::ios::out | std::ios::trunc | std::ios::binary);
		if (writestr.is_open())
		{
			onnx::ModelProto model;
			google::protobuf::util::JsonParseOptions options;
			options.ignore_unknown_fields = true;
			if (google::protobuf::util::Status::OK !=
				google::protobuf::util::JsonStringToMessage(
					jsonstr, &model, options))
			{
				teq::fatal("failed to parse json model");
			}
			if (false == model.SerializeToOstream(&writestr))
			{
				teq::fatalf("failed to serialize protobuf to %s",
					writepath.c_str());
			}
			writestr.close();
			return;
		}
	}
	else
	{
		std::ofstream writestr(writepath);
		if (writestr.is_open())
		{
			writestr << jsonstr;
			writestr.flush();
			writestr.close();
			return;
		}
	}
	// write to stdout when all options fail
	std::cout << jsonstr << std::endl;
}

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

	teq::get_logger().set_log_level(teq::INFO);

	std::ifstream readstr(readpath);
	if (readstr.is_open())
	{
		std::string jsonstr;
		if (estd::has_affix(readpath, onnx_ext))
		{
			onnx::ModelProto model;
			if (false == model.ParseFromIstream(&readstr))
			{
				teq::fatalf("failed to parse from istream when read file %s",
					readpath.c_str());
			}
			google::protobuf::util::JsonPrintOptions options;
			options.add_whitespace = true;
			options.always_print_primitive_fields = true;
			if (google::protobuf::util::Status::OK !=
				google::protobuf::util::MessageToJsonString(
					model, &jsonstr, options))
			{
				teq::fatal("failed to parse onnx model");
			}
		}
		else
		{
			std::stringstream ss;
			ss << readstr.rdbuf();
			jsonstr = ss.str();
		}

		write_json(writepath, jsonstr);

		readstr.close();
	}
	else
	{
		teq::warnf("failed to read file `%s`", readpath.c_str());
	}

	return 0;
}
