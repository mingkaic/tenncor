#include <fstream>

#include <google/protobuf/util/json_util.h>

#include "flag/flag.hpp"
#include "fmts/fmts.hpp"

#include "pbm/graph.pb.h"

const std::string pbx_ext = ".pbx";

bool has_affix (const std::string& str, const std::string& affix)
{
	size_t n = str.size();
	size_t naffix = affix.size();
	if (n < naffix)
	{
		return false;
	}
	return std::equal(str.begin() + n - naffix,
		str.end(), affix.begin());
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
		if (has_affix(readpath, pbx_ext))
		{
			cortenn::Graph graph;
			if (false == graph.ParseFromIstream(&readstr))
			{
				logs::fatalf("failed to parse from istream when read file %s",
					readpath.c_str());
			}
			google::protobuf::util::JsonPrintOptions options;
			options.add_whitespace = true;
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

		std::ofstream writestr(writepath);
		if (writestr.is_open())
		{
			if (has_affix(writepath, pbx_ext))
			{
				cortenn::Graph graph;
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
			}
			else
			{
				writestr << jsonstr;
				writestr.flush();
			}
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
