#include <iomanip>
#include <sstream>
#include <regex>

#include "nlohmann/json.hpp"

#include "err/log.hpp"

#include "age/parser.hpp"

#ifdef AGE_PARSER_HPP

static const std::string derivative_name = "arg0";

static const std::string special_repl = "$!";

static const std::string opcode_enum = "_GENERATED_OPCODES";

static const std::string grad_signature =
	"ade::Tensorptr %s<%s> (ade::ArgsT args, size_t idx)";

void find_replace (std::string& data,
	std::string target, std::string replacement)
{
	size_t pos = data.find(target);
	while (pos != std::string::npos)
	{
		// Replace this occurrence of Sub String
		data.replace(pos, target.size(), replacement);
		// Get the next occurrence from the current position
		pos = data.find(target, pos + target.size());
	}
}

Func parse_derive (std::sring opcode, std::string derive)
{
	find_replace(derive, special_repl, "idx");
	find_replace(derive, "$@", "args");
	std::regex_replace(derive, std::regex("\\$(\\d+)"), "args[$1]");
	std::string signature = err::sprintf(grad_signature,
		varname, opcode.c_str());
	return Func{signature, derive};
}

std::string read (nlohmann::json::object_t refmap, nlohmann::json ref)
{
	if (nlohmann::json::value_t::string == ref.type())
	{
		return ref;
	}
	else if (nlohmann::json::value_t::object != ref.type())
	{
		std::stringstream ss;
		ss << std::setw(4) << ref;
		err::fatalf("cannot read %s to as string", ss.str().c_str());
	}
	nlohmann::json::object_t obj = ref;
	auto it = obj.find("ref");
	if (obj.end() != it)
	{
		std::string key = it->second;
		auto refit = refmap.find(key);
		if (refmap.end() == refit)
		{
			err::fatalf("cannot find ref %s", key.c_str());
		}
		return refit->second;
	}
	std::string format = read(refmap, ref.at("format"));
	nlohmann::json::object_t argmap = ref.at("args");
	for (auto& arg : argmap)
	{
		std::string target = arg.first;
		std::string replacement = arg.second;
		find_replace(format, "$" + target, replacement);
	}
	return format;
}

void unmarshal_json (File& out, std::istream& jstr)
{
	try
	{
		nlohmann::json json_rep;
		jstr >> json_rep;
		// expect root representation to always be json object
		if (nlohmann::json::value_t::object != json_rep.type())
		{
			std::stringstream ss;
			ss << std::setw(4) << json_rep;
			err::fatalf("cannot parse non-object json representation: %s",
				ss.str().c_str());
		}
		// optionals
		out.includes_ = json_rep.value("includes", out.includes_);

		nlohmann::json::object_t refs;
		refs = json_rep.value("refs", refs);

		// apis is manditory
		nlohmann::json::object_t opcodes;
		opcodes = json_rep.value("opcodes", opcodes);
		StringsT opnum;
		for (auto& opcode : opcodes)
		{
			opnum.push_back(opcode.first);
			out.funcs_.push_back(
				parse_derive(opcode.first, read(refs, opcode.second)));
		}
		out.enums_.push_back(Enum{opcode_enum, opnum});

		nlohmann::json::array_t apis = json_rep.at("apis");
		for (nlohmann::json::object_t api : apis)
		{
			out.funcs_.push_back(Func{
				read(refs, api.at("io")),
				read(refs, api.at("out")),
			});
		}
	}
	catch (std::exception& exc)
	{
		err::fatal(exc.what());
	}
}

#endif
