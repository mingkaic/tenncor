#include <regex>

#include "experimental/opt/rule/convert.hpp"

#ifndef OPT_RULE_PARSE_HPP
#define OPT_RULE_PARSE_HPP

namespace opt
{

namespace rule
{

const std::regex comment_pattern("[^\\\\]?(//.*)");

const std::regex symbol_pattern("symbol (\\w+)$");

const std::regex convert_pattern("(.+)\\s*=>\\s*(.+)$");

const std::regex func_pattern("((?:group:)?\\w+)\\((.+)\\)$");

const std::regex variadic_pattern("\\.\\.(\\w+)");

WriterptrT make_writer (std::string source,
	const std::unordered_map<std::string,size_t>& symbols)
{
	fmts::trim(source);
	auto it = symbols.find(source);
	if (symbols.end() != it)
	{
		// is a variable
		return std::make_shared<AnyRule>(it->second);
	}
	std::smatch sm;
	if (std::regex_match(source, sm, func_pattern))
	{
		// is a function
		std::string funcname = sm[1];
		std::string argstr = sm[2];
		std::string arg;
		std::vector<std::string> args;
		size_t levels = 0;
		for (char c : argstr)
		{
			switch (c)
			{
				case ',':
					if (0 == levels)
					{
						args.push_back(arg);
						arg = "";
						continue;
					}
					break;
				case '(':
					++levels;
					break;
				case ')':
					--levels;
					break;
			}
			arg.push_back(c);
		}
		size_t variadic_id = std::string::npos;
		if (arg.size() > 0)
		{
			std::smatch sm;
			if (std::regex_match(arg, sm, variadic_pattern))
			{
				auto it = symbols.find(sm[1]);
				if (symbols.end() == it)
				{
					logs::fatalf("unknown variadic variable %s", arg.c_str());
				}
				variadic_id = it->second;
			}
			else
			{
				args.push_back(arg);
			}
		}
		WriterArgsT rule_args;
		rule_args.reserve(args.size());
		for (std::string& arg : args)
		{
            // todo: parse non-identity rules
			rule_args.push_back(WriterArg{
				make_writer(arg, symbols),
				ade::identity,
				ade::CoordptrT(),
			});
		}
		if (variadic_id < symbols.size())
		{
			return std::make_shared<VariadicFunc>(
                funcname, rule_args, variadic_id);
		}
		else
		{
			return std::make_shared<Func>(funcname, rule_args);
		}
	}
	std::istringstream iss(source);
	double num;
	iss >> std::noskipws >> num;
	if (false == iss.eof() || iss.fail())
	{
		logs::fatalf("unknown symbol %s", source.c_str());
	}
	// is a scalar
	return std::make_shared<Scalar>(num);
}

BuilderT make_builder (std::string dest,
	const std::unordered_map<std::string,size_t>& symbols)
{
	fmts::trim(dest);
	auto it = symbols.find(dest);
	// if (symbols.end() != it)
	// {
	// 	// is a variable
	// 	return std::make_shared<VarTarget<T>>(it->second);
	// }
	// std::smatch sm;
	// if (std::regex_match(dest, sm, func_pattern))
	// {
	// 	// is a function
	// 	std::string funcname = sm[1];
	// 	std::string argstr = sm[2];
	// 	std::vector<std::string> args;
	// 	{
	// 		std::string arg;
	// 		size_t levels = 0;
	// 		for (char c : argstr)
	// 		{
	// 			switch (c)
	// 			{
	// 				case ',':
	// 					if (0 == levels)
	// 					{
	// 						args.push_back(arg);
	// 						arg = "";
	// 						continue;
	// 					}
	// 					break;
	// 				case '(':
	// 					++levels;
	// 					break;
	// 				case ')':
	// 					--levels;
	// 					break;
	// 			}
	// 			arg.push_back(c);
	// 		}
	// 		args.push_back(arg);
	// 	}
	// 	TargetArgsT<T> tar_args;
	// 	std::vector<size_t> variadics;
	// 	tar_args.reserve(args.size());
	// 	variadics.reserve(args.size());
	// 	for (std::string& arg : args)
	// 	{
	// 		std::smatch sm;
	// 		if (std::regex_match(arg, sm, variadic_pattern))
	// 		{
	// 			auto it = symbols.find(sm[1]);
	// 			if (symbols.end() == it)
	// 			{
	// 				logs::fatalf("unknown variadic variable %s", arg.c_str());
	// 			}
	// 			variadics.push_back(it->second);
	// 		}
	// 		else
	// 		{
	// 			tar_args.push_back(TargetArg<T>{
	// 				make_builder<T>(arg, symbols),
	// 				ade::identity,
	// 				ade::CoordptrT(),
	// 			});
	// 		}
	// 	}
	// 	return std::make_shared<FuncTarget<T>>(
	// 		ade::Opcode{funcname, age::get_op(funcname)},
	// 		tar_args, variadics);
	// }
	// std::istringstream iss(dest);
	// double num;
	// iss >> std::noskipws >> num;
	// if (false == iss.eof() || iss.fail())
	// {
	// 	logs::fatalf("unknown symbol %s", dest.c_str());
	// }
	// // is a scalar
	// return std::make_shared<ScalarTarget<T>>(num);
}

static std::string strip_comment (std::string line)
{
	std::smatch sm;
	if (std::regex_search(line, sm, comment_pattern))
	{
		size_t comment_begin = line.size() - sm[1].length();
		line = line.substr(0, comment_begin);
	}
	return line;
}

ConversionsT parse (std::istream& is)
{
	ConversionsT conversions;
	std::unordered_map<std::string,size_t> symbols;
	std::string line;
	while (std::getline(is, line))
	{
		line = strip_comment(line);
		if (line.empty())
		{
			continue;
		}
        fmts::trim(line);
		std::smatch sm;
		if (std::regex_match(line, sm, symbol_pattern))
		{
			std::string symbol = sm[1];
			symbols.emplace(symbol, symbols.size());
		}
		else if (std::regex_match(line, sm, convert_pattern))
		{
			std::string source = sm[1];
			std::string dest = sm[2];
			WriterptrT rule_src = make_writer(source, symbols);
			BuilderT rule_dest = make_builder(dest, symbols);
			conversions.push_back(std::make_shared<Conversion>(
                line,
                rule_src,
                rule_dest));
		}
		else
		{
			logs::warnf("invalid statement %s", line.c_str());
		}
	}
	return conversions;
}

ConversionsT get_configs (void)
{
	ConversionsT rules;
	std::ifstream config_stream("ead/cfg/optimizations.rules");
	if (config_stream.is_open())
	{
		rules = parse(config_stream);
	}
	else
	{
		logs::warn("optimization configuration not found");
	}
	return rules;
}

}

}

#endif // OPT_RULE_PARSE_HPP
