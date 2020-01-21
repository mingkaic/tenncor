#include "teq/logs.hpp"

#ifdef TEQ_LOGS_HPP

namespace teq
{

#define LOG_DEFN(LOG_LEVEL){\
	auto logger = static_cast<logs::iLogger*>(\
		config::global_config.get_obj(logger_key));\
	if (nullptr == logger)\
	{\
		logs::error("failed to log: " + msg);\
		return;\
	}\
	if (logger->supports_level(LOG_LEVEL))\
	{ logger->log(LOG_LEVEL, msg); }\
}

std::string get_log_level (void)
{
	auto logger = static_cast<logs::iLogger*>(
		config::global_config.get_obj(logger_key));
	if (nullptr == logger)
	{
		logs::error("failed to get log level");
		return "fatal";
	}
	return logger->get_log_level();
}

void set_log_level (const std::string& log_level)
{
	auto logger = static_cast<logs::iLogger*>(
		config::global_config.get_obj(logger_key));
	if (nullptr == logger)
	{
		logs::error("failed to set log level");
		return;
	}
	if (logger->supports_level(log_level))
	{
		logger->set_log_level(log_level);
	}
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

}

#endif
