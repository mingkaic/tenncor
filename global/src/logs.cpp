
#include "global/g3logs.hpp"
#include "global/logs.hpp"

#ifdef GLOBAL_LOGS_HPP

namespace global
{

const std::string logger_key = "logger";

void set_logger (logs::iLogger* logger, global::CfgMapptrT ctx)
{
	ctx->rm_entry(global::logger_key);
	if (logger)
	{
		ctx->template add_entry<logs::iLogger>(global::logger_key,
			[=](){ return logger; });
	}
}

logs::iLogger& get_logger (const global::CfgMapptrT& ctx)
{
	auto log = static_cast<logs::iLogger*>(
		ctx->get_obj(global::logger_key));
	if (nullptr == log)
	{
		log = new G3Logger();
		set_logger(log, ctx);
	}
	return *log;
}

void set_log_level (const std::string& log_level, global::CfgMapptrT ctx)
{
	auto& logger = get_logger(ctx);
	if (logger.supports_level(log_level))
	{
		logger.set_log_level(log_level);
	}
}

std::string get_log_level (const global::CfgMapptrT& ctx)
{
	return get_logger(ctx).get_log_level();
}

#define LOG_DEFN(LOG_LEVEL){\
	auto& logger = global::get_logger();\
	if (logger.supports_level(LOG_LEVEL))\
	{ logger.log(LOG_LEVEL, msg); }\
}

void debug (const std::string& msg)
LOG_DEFN(logs::debug_level)

void info (const std::string& msg)
LOG_DEFN(logs::info_level)

void warn (const std::string& msg)
LOG_DEFN(logs::warn_level)

void error (const std::string& msg)
LOG_DEFN(logs::error_level)

void throw_err (const std::string& msg)
LOG_DEFN(global::throw_err_level)

void fatal (const std::string& msg)
LOG_DEFN(logs::fatal_level)

#undef LOG_DEFN

}

#endif
