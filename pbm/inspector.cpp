#include <fstream>

#include <google/protobuf/util/json_util.h>

#include "flag/flag.hpp"
#include "fmts/fmts.hpp"
#include "estd/strs.hpp"

#include "pbm/graph.pb.h"

const std::string pbx_ext = ".pbx";

void write_json (std::string writepath, const std::string& jsonstr)
{
	if (estd::has_affix(writepath, pbx_ext))
	{
		std::fstream writestr(writepath, std::ios::out | std::ios::trunc | std::ios::binary);
		if (writestr.is_open())
		{
			tenncor::Graph graph;
			google::protobuf::util::JsonParseOptions options;
			options.ignore_unknown_fields = true;
			if (google::protobuf::util::Status::OK !=
				google::protobuf::util::JsonStringToMessage(
					jsonstr, &graph, options))
			{
				logs::fatal("failed to parse json graph");
			}
			if (false == graph.SerializeToOstream(&writestr))
			{
				logs::fatalf("failed to serialize protobuf to %s",
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

	logs::get_logger().set_log_level(logs::INFO);

	std::ifstream readstr(readpath);
	if (readstr.is_open())
	{
		std::string jsonstr;
		if (estd::has_affix(readpath, pbx_ext))
		{
			tenncor::Graph graph;
			if (false == graph.ParseFromIstream(&readstr))
			{
				logs::fatalf("failed to parse from istream when read file %s",
					readpath.c_str());
			}
			google::protobuf::util::JsonPrintOptions options;
			options.add_whitespace = true;
			options.always_print_primitive_fields = true;
			if (google::protobuf::util::Status::OK !=
				google::protobuf::util::MessageToJsonString(
					graph, &jsonstr, options))
			{
				logs::fatal("failed to parse pb graph");
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
		logs::warnf("failed to read file `%s`", readpath.c_str());
	}

	return 0;
}
