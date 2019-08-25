#include <iostream>
#include <fstream>
#include <iomanip>

#include "flag/flag.hpp"
#include "fmts/fmts.hpp"

#include "pbm/graph.pb.h"

int main (int argc, const char** argv)
{
	std::string readpath;
	flag::FlagSet flags("inspector");
	flags.add_flags()
		("read", flag::opt::value<std::string>(&readpath),
			"filename of model to inspect");

	if (false == flags.parse(argc, argv))
	{
		return 1;
	}

	std::ifstream readstr(readpath);
	if (readstr.is_open())
	{
		cortenn::Graph graph;
		if (false == graph.ParseFromIstream(&readstr))
		{
			logs::fatalf("failed to parse from istream when read file %s",
				readpath.c_str());
		}
		std::cout
			<< "{\n"
			<< "\t\"label\":\"" << graph.label() << "\",\n"
			<< "\t\"nodes\":[\n";
		auto& nodes = graph.nodes();
		for (auto& node : nodes)
		{
			std::cout << "\t\t{\n";

			auto labels = node.labels();
			for (auto& label : labels)
			{
				label = fmts::sprintf("\"%s\"", label.c_str());
			}
			std::cout << "\t\t\t\"labels\":["
				<< fmts::join(",", labels.begin(), labels.end()) << "],\n";
			auto tags = node.tags();
			std::cout << "\t\t\t\"tags\":{\n";
			for (auto& tag : tags)
			{
				auto tag_label = tag.second.labels();
				for (auto& label : tag_label)
				{
					label = fmts::sprintf("\"%s\"", label.c_str());
				}
				std::cout << fmts::sprintf("\t\t\t\t\"%s\":[%s],\n",
					tag.first.c_str(),
					fmts::join(",", tag_label.begin(),
						tag_label.end()).c_str());
			}
			std::cout << "\t\t\t}\n";

			std::cout << "\t\t},\n";
		}
		std::cout
			<< "\t]\n"
			<< "}\n";
		readstr.close();
	}
	else
	{
		logs::warnf("failed to read file `%s`", readpath.c_str());
	}

	return 0;
}
