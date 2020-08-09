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
		log = new logs::DefLogger();
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

void trace (const std::string& msg)
LOG_DEFN("trace")

void debug (const std::string& msg)
LOG_DEFN("debug")

void info (const std::string& msg)
LOG_DEFN("info")

void warn (const std::string& msg)
LOG_DEFN("warn")

void error (const std::string& msg)
LOG_DEFN("error")

void fatal (const std::string& msg)
LOG_DEFN("fatal")

#undef LOG_DEFN

}

#endif
