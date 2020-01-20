#include "teq/config.hpp"

#ifdef TEQ_CONFIG_HPP

namespace teq
{

std::shared_ptr<estd::iConfig> global_config = nullptr;

#define LOG_DEFN(LEVEL){\
	if (nullptr == global_config)\
	{\
		logs::error("failed to process " + msg);\
		return;\
	}\
	auto logger = static_cast<logs::iLogger*>(\
		global_config->get_obj(logger_key));\
	logger->log(LEVEL, msg);\
}

/// Log at trace level
void trace (std::string msg)
LOG(logs::TRACE)

/// Log at debug level
void debug (std::string msg)
LOG(logs::DEBUG)

/// Log at info level
void info (std::string msg)
LOG(logs::INFO)

/// Log at warn level
void warn (std::string msg)
LOG(logs::WARN)

/// Log at error level
void error (std::string msg)
LOG(logs::ERROR)

/// Log at fatal level
void fatal (std::string msg)
LOG(logs::FATAL)

}

#endif
