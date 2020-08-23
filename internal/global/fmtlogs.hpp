
#ifndef GLOBAL_FMTLOGS_HPP
#define GLOBAL_FMTLOGS_HPP

#include "logs/logs.hpp"

namespace global
{

struct FormatLogger final : public logs::iLogger
{
	FormatLogger (logs::iLogger* sublogger,
		const std::string& prefix = "",
		const std::string& affix = "") :
		sublogger_(sublogger), prefix_(prefix), affix_(affix) {}

	/// Implementation of iLogger
	std::string get_log_level (void) const override
	{
		return sublogger_->get_log_level();
	}

	/// Implementation of iLogger
	void set_log_level (const std::string& log_level) override
	{
		sublogger_->set_log_level(log_level);
	}

	/// Implementation of iLogger
	bool supports_level (size_t msg_level) const override
	{
		return sublogger_->supports_level(msg_level);
	}

	/// Implementation of iLogger
	bool supports_level (const std::string& msg_level) const override
	{
		return sublogger_->supports_level(msg_level);
	}

	/// Implementation of iLogger
	void log (size_t msg_level, const std::string& msg) override
	{
		sublogger_->log(msg_level, prefix_ + msg + affix_);
	}

	/// Implementation of iLogger
	void log (const std::string& msg_level, const std::string& msg) override
	{
		sublogger_->log(msg_level, prefix_ + msg + affix_);
	}

	logs::iLogger* sublogger_;

	std::string prefix_;

	std::string affix_;
};

}

#endif // GLOBAL_FMTLOGS_HPP
