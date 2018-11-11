#include <sstream>
#include <regex>

#include "nlohmann/json.hpp"

#include "err/log.hpp"

#include "age/parser.hpp"

#ifdef AGE_PARSER_HPP

static const std::string opcode_enum = "_GENERATED_OPCODES";

static const std::string data_signature = "template <typename T>"
	"ade::Tensor* data (T scalar, ade::Shape shape)";

static const std::string sum_opcode_signature =
	"ade::Opcode sum_opcode (void)";

static const std::string prod_opcode_signature =
	"ade::Opcode prod_opcode (void)";

static const std::string map_name = "code2name";

static const std::string nameop_signature =
	opcode_enum + " nameop (std::string name)";

static const std::string namemap_call = map_name + ".find(name)->second";

static const std::string opname_switches_signature =
	"std::string opname (" + opcode_enum + " code)";

static const std::string grad_switches_signature =
	"ade::Tensorptr grad_rule (size_t code, TensT args, size_t idx)";

void find_replace (std::string& data,
	std::string target, std::string replacement)
{
	size_t pos = data.find(target);
	while (pos != std::string::npos)
	{
		// Replace this occurrence of Sub String
		data.replace(pos, target.size(), replacement);
		// Get the next occurrence from the current position
		pos = data.find(target, pos + replacement.size());
	}
}

std::string parse_derive (std::string derive)
{
	find_replace(derive, "$@", "args");
	return std::regex_replace(derive, std::regex("\\$(\\d+)"), "args[$1]");
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

void unmarshal_json (File& runtime_file, File& api_file,
	std::istream& jstr)
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
		StringsT includes;
		includes = json_rep.value("includes", includes);

		nlohmann::json::object_t refs;
		refs = json_rep.value("refs", refs);

		// runtime implementation is manditory
		runtime_file.includes_.insert(
			runtime_file.includes_.end(),
			includes.begin(), includes.end());
		nlohmann::json::object_t runtime;
		runtime = json_rep.value("runtime", runtime);

		std::string data_ret = runtime.at("data");
		std::string sum_opcode_ret = runtime.at("sum_opcode");
		std::string prod_opcode_ret = runtime.at("prod_opcode");
		runtime_file.funcs_.push_back(Func{
			data_signature,
			{ std::make_shared<ReturnStmt>(data_ret) }, true
		});
		runtime_file.funcs_.push_back(Func{
			sum_opcode_signature,
			{ std::make_shared<ReturnStmt>(sum_opcode_ret) },
		});
		runtime_file.funcs_.push_back(Func{
			prod_opcode_signature,
			{ std::make_shared<ReturnStmt>(prod_opcode_ret) },
		});

		nlohmann::json::object_t opcodes = runtime.at("opcodes");
		StringsT opnum;
		std::unordered_map<std::string,std::string> name_map;
		auto grad_switches = new SwitchStmt("code");
		auto opname_switches = new SwitchStmt("code");
		for (auto& opcode : opcodes)
		{
			std::string opstr = "\"" + opcode.first + "\"";
			opnum.push_back(opcode.first);
			name_map[opstr] = opcode.first;
			grad_switches->cases_[opcode.first] =
				std::make_shared<ReturnStmt>(parse_derive(opcode.second));
			opname_switches->cases_[opcode.first] =
				std::make_shared<ReturnStmt>(opstr);
		}
		runtime_file.maps_.push_back(MapRep{map_name, {"std::string", opcode_enum}, name_map, true});
		runtime_file.enums_.push_back(Enum{opcode_enum, opnum});
		runtime_file.funcs_.push_back(Func{
			nameop_signature,
			{ std::make_shared<ReturnStmt>(namemap_call) },
		});
		runtime_file.funcs_.push_back(Func{
			opname_switches_signature,
			{ StmtptrT(opname_switches) },
		});
		runtime_file.funcs_.push_back(Func{
			grad_switches_signature,
			{ StmtptrT(grad_switches) },
		});

		// apis implementation is manditory
		api_file.includes_.insert(
			api_file.includes_.end(),
			includes.begin(), includes.end());
		nlohmann::json::array_t apis = json_rep.at("apis");
		for (nlohmann::json::object_t api : apis)
		{
			api_file.funcs_.push_back(Func{
				read(refs, api.at("io")),
				{ std::make_shared<ReturnStmt>(read(refs, api.at("out"))) },
			});
		}
	}
	catch (std::exception& exc)
	{
		err::fatal(exc.what());
	}
}

#endif
