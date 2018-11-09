#include <fstream>
#include <iostream>
#include <string>
#include <regex>
#include <unordered_map>

#include "err/log.hpp"

#include "age/parser.hpp"

const char* program_name = "agen";

std::regex re_flag("-(\\w+)");

struct Flags
{
	void parse (std::string key,
		std::string* out, std::string description)
	{
		flags_[key] = out;

		args_.push_back(err::sprintf("[-%s <string>]", key.c_str()));
		descriptions_.push_back(description);
	}

	bool process (size_t argc, char** argv)
	{
		std::string flag;
		for (size_t i = 1; i < argc; i += 2)
		{
			if (is_flag(flag, argv[i]))
			{
				if (std::string* out = check_flag(flag))
				{
					out->assign(argv[i + 1]);
				}
				else
				{
					help();
					return false;
				}
			}
			else
			{
				help();
				return false;
			}
		}
		return true;
	}

private:
	void help (void) const
	{
		size_t strn = 0;
		std::cerr << "usage: " << program_name;
		for (std::string arg : args_)
		{
			strn = std::max(arg.size(), strn);
			std::cerr << " " << arg;
		}
		std::cerr << "\n";
		for (size_t i = 0, n = descriptions_.size(); i < n; ++i)
		{
			size_t npad = strn - args_[i].size();
			std::cerr << "    " << args_[i] << std::string(npad, ' ') <<
				"    " << descriptions_[i] << "\n";
		}
	}

	bool is_flag (std::string& flag, char* arg) const
	{
		std::smatch matches;
		bool matched = std::regex_match(flag, matches, re_flag);
		if (matched)
		{
			flag = matches[1];
		}
		return matched;
	}

	std::string* check_flag (std::string key)
	{
		auto it = flags_.find(key);
		if (flags_.end() != it)
		{
			return it->second;
		}
		return nullptr;
	}

	StringsT args_;
	StringsT descriptions_;

	std::unordered_map<std::string,std::string*> flags_;
};

std::string strip_header (std::string header, std::string prefix)
{
	size_t prefix_size = prefix.size();
	if (prefix_size > 0 && header.size() > prefix_size &&
		0 == prefix.compare(header.substr(0, prefix_size)))
	{
		// only strip if header prefix exists
		return header.substr(prefix_size);
	}
	return header;
}

int main (int argc, char** argv)
{
	std::string cfgpath;
	std::string headerpath;
	std::string sourcepath;
	std::string strip_prefix;

	Flags flags;
	flags.parse("cfg", &cfgpath,
		"path to configuration json file (default: read from stdin)");
	flags.parse("hdr", &headerpath,
		"path to header file to write (default: write to stdout)");
	flags.parse("src", &sourcepath,
		"path to source file to write (default: write to stdout)");
	flags.parse("strip_hdr", &strip_prefix,
		"strip prefix from hdr to include in src (default: strip nothing)");
	if (false == flags.process(argc, argv))
	{
		return 1;
	}

	std::ifstream instr(cfgpath);

	File file;
	if (instr.is_open())
	{
		unmarshal_json(file, instr);
		instr.close();
	}
	else
	{
		unmarshal_json(file, std::cin);
	}

	std::ofstream header(headerpath);
	if (header.is_open())
	{
		file.decl(header);
		header.close();
	}
	else
	{
		file.decl(std::cout);
	}

	std::ofstream source(sourcepath);
	std::string includepath = strip_header(headerpath, strip_prefix);
	if (source.is_open())
	{
		source << "#include \"" << includepath << "\"\n\n";
		file.defn(source);
	}
	else
	{
		std::cout << "#include \"" << includepath << "\"\n\n";
		file.defn(std::cout);
	}

	return 0;
}
