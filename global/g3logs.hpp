
// #include <source_location>

#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>

#include "logs/logs.hpp"

#include "estd/contain.hpp"

#ifndef GLOBAL_G3LOGGER_HPP
#define GLOBAL_G3LOGGER_HPP

namespace std // mock source location until cpp 20 is implemented
{

struct source_location
{
	static source_location current (void) { return source_location(); }

	const char* file_name (void) const noexcept { return ""; }

	std::uint32_t line (void) const noexcept { return 0; }

	const char* function_name (void) const noexcept { return ""; }
};

}

namespace g3
{

static const int kErrorValue = 750;

static const int kThrowErrValue = 751;

}

const LEVELS ERROR{g3::kErrorValue, {"ERROR"}};

const LEVELS THROW_ERR{g3::kThrowErrValue, {"THROW_ERR"}};

namespace global
{

const std::string app_name = "tenncor";

const std::string throw_err_level = "throw_err";

static const types::StrUMapT<LEVELS> str2lvl = {
	{logs::debug_level, DBUG},
	{logs::info_level, INFO},
	{logs::warn_level, WARNING},
	{logs::error_level, ERROR},
	{throw_err_level, THROW_ERR},
	{logs::fatal_level, FATAL},
};

static const std::unordered_map<int,LEVELS> int2lvl = {
	{DBUG.value, DBUG},
	{INFO.value, INFO},
	{WARNING.value, WARNING},
	{ERROR.value, ERROR},
	{THROW_ERR.value, THROW_ERR},
	{FATAL.value, FATAL},
};

using G3WorkerT = std::unique_ptr<g3::LogWorker>;

using ProcG3WorkerF = std::function<void(G3WorkerT&)>;

void add_stdio_sink (G3WorkerT& worker);

#define CUSTOM_LOG(file, func, line, level)\
if(g3::logLevel(level)) LogCapture(file, line, func, level).stream()

struct G3Logger final : public logs::iLogger
{
	G3Logger (ProcG3WorkerF add_sink = add_stdio_sink) :
		worker_(g3::LogWorker::createLogWorker()),
		level_(INFO)
	{
		add_sink(worker_);
		g3::initializeLogging(worker_.get());
		set_log_level(logs::info_level);
	}

	~G3Logger (void)
	{
		g3::internal::shutDownLogging();
	}

	/// Implementation of iLogger
	std::string get_log_level (void) const override
	{
		return level_.text;
	}

	/// Implementation of iLogger
	void set_log_level (const std::string& log_level) override
	{
		estd::get(level_, str2lvl, log_level);
		g3::log_levels::setHighest(level_);
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
	void log (size_t msg_level, const std::string& msg) override
	{
		if (estd::has(int2lvl, msg_level))
		{
			log(int2lvl.at(msg_level), msg);
		}
	}

	/// Implementation of iLogger
	void log (const std::string& msg_level, const std::string& msg) override
	{
		if (estd::has(str2lvl, msg_level))
		{
			log(str2lvl.at(msg_level), msg);
		}
	}

	void log (LEVELS msg_level, const std::string& msg,
		const std::source_location& location = std::source_location::current())
	{
		if (g3::logLevel(msg_level))
		{
			LogCapture(location.file_name(), location.line(),
				location.function_name(), msg_level).stream() << msg;
		}
	}

	G3WorkerT worker_;

	LEVELS level_;
};

}

#endif // GLOBAL_G3LOGGER_HPP
