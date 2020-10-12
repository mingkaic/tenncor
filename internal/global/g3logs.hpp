
#ifndef GLOBAL_G3LOGGER_HPP
#define GLOBAL_G3LOGGER_HPP

// #include <source_location>

#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>

#include "logs/logs.hpp"

#include "estd/estd.hpp"

namespace g3
{

static const int kErrorValue = WARNING.value + 1;

static const int kThrowErrValue = WARNING.value + 2;

}

const LEVELS ERROR{g3::kErrorValue, {"ERROR"}};

const LEVELS THROW_ERR{g3::kThrowErrValue, {"THROW_ERR"}};

namespace global
{

static const types::StrUMapT<LEVELS> str2lvl = {
	{logs::debug_level, DBUG},
	{logs::info_level, INFO},
	{logs::warn_level, WARNING},
	{logs::error_level, ERROR},
	{logs::throw_err_level, THROW_ERR},
	{logs::fatal_level, FATAL},
	{DBUG.text, DBUG},
	{INFO.text, INFO},
	{WARNING.text, WARNING},
	{ERROR.text, ERROR},
	{THROW_ERR.text, THROW_ERR},
	{FATAL.text, FATAL},
};

static const std::unordered_map<int,LEVELS> int2lvl = {
	{DBUG.value, DBUG},
	{INFO.value, INFO},
	{WARNING.value, WARNING},
	{ERROR.value, ERROR},
	{THROW_ERR.value, THROW_ERR},
	{FATAL.value, FATAL},
};

static const std::unordered_map<int,std::string> int2str = {
	{DBUG.value, logs::debug_level},
	{INFO.value, logs::info_level},
	{WARNING.value, logs::warn_level},
	{ERROR.value, logs::error_level},
	{THROW_ERR.value, logs::throw_err_level},
	{FATAL.value, logs::fatal_level},
};

using G3WorkerT = std::unique_ptr<g3::LogWorker>;

using ProcG3WorkerF = std::function<void(G3WorkerT&)>;

void add_stdio_sink (G3WorkerT& worker, std::ostream& outs = std::cout);

#define CUSTOM_LOG(file, func, line, level)\
if(g3::logLevel(level)) LogCapture(file, line, func, level).stream()

struct G3Logger final : public logs::iLogger
{
	G3Logger (ProcG3WorkerF add_sink =
		[](G3WorkerT& worker){ add_stdio_sink(worker); }) :
		worker_(g3::LogWorker::createLogWorker()),
		level_(INFO)
	{
		// add custom log levels
		g3::only_change_at_initialization::addLogLevel(ERROR, true);
		g3::only_change_at_initialization::addLogLevel(THROW_ERR, true);
		// g3 initialization
		add_sink(worker_);
		g3::initializeLogging(worker_.get());
		// set default log level
		set_log_level(logs::info_level);
	}

	~G3Logger (void)
	{
		g3::internal::shutDownLogging();
	}

	/// Implementation of iLogger
	std::string get_log_level (void) const override
	{
		return int2str.at(level_.value);
	}

	/// Implementation of iLogger
	void set_log_level (const std::string& log_level) override
	{
		if (estd::get(level_, str2lvl, log_level))
		{
			g3::log_levels::setHighest(level_);
		}
	}

	/// Implementation of iLogger
	bool supports_level (size_t msg_level) const override
	{
		return estd::has(int2lvl, msg_level);
	}

	/// Implementation of iLogger
	bool supports_level (const std::string& msg_level) const override
	{
		return estd::has(str2lvl, msg_level);
	}

	/// Implementation of iLogger
	void log (size_t msg_level, const std::string& msg,
		const logs::SrcLocT& location = logs::SrcLocT::current()) override
	{
		if (estd::has(int2lvl, msg_level))
		{
			log(int2lvl.at(msg_level), msg, location);
		}
	}

	/// Implementation of iLogger
	void log (const std::string& msg_level, const std::string& msg,
		const logs::SrcLocT& location = logs::SrcLocT::current()) override
	{
		if (estd::has(str2lvl, msg_level))
		{
			log(str2lvl.at(msg_level), msg, location);
		}
	}

	void log (LEVELS msg_level, const std::string& msg,
		const logs::SrcLocT& location = logs::SrcLocT::current())
	{
		if (g3::logLevel(msg_level))
		{
			LogCapture(location.file_name(), location.line(),
				location.function_name(), msg_level).stream() << msg;
			if (msg_level.value >= g3::kThrowErrValue)
			{
				throw std::runtime_error(msg);
			}
		}
	}

	G3WorkerT worker_;

	LEVELS level_;
};

}

#endif // GLOBAL_G3LOGGER_HPP
