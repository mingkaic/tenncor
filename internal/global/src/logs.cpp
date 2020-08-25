
#include "internal/global/logs.hpp"

#ifdef GLOBAL_LOGS_HPP

namespace global
{

const std::string logger_key = "logger";

void set_logger (logs::iLogger* logger, CfgMapptrT ctx)
{
	ctx->rm_entry(logger_key);
	if (logger)
	{
		ctx->template add_entry<logs::iLogger>(logger_key,
			[=](){ return logger; });
	}
}

logs::iLogger& get_logger (const CfgMapptrT& ctx)
{
	auto log = static_cast<logs::iLogger*>(
		ctx->get_obj(logger_key));
	if (nullptr == log)
	{
		log = new G3Logger();
		set_logger(log, ctx);
	}
	return *log;
}

void set_log_level (const std::string& log_level, CfgMapptrT ctx)
{
	auto& logger = get_logger(ctx);
	if (logger.supports_level(log_level))
	{
		logger.set_log_level(log_level);
	}
}

std::string get_log_level (const CfgMapptrT& ctx)
{
	return get_logger(ctx).get_log_level();
}

#define LOG_DEFN(LOG_LEVEL){\
	auto& logger = get_logger();\
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

[[ noreturn ]]
void throw_err (const std::string& msg)
LOG_DEFN(logs::throw_err_level)

[[ noreturn ]]
void fatal (const std::string& msg)
LOG_DEFN(logs::fatal_level)

#undef LOG_DEFN

}

#endif
