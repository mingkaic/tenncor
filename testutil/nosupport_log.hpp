
#ifndef TEST_NOSUPPORT_HPP
#define TEST_NOSUPPORT_HPP

#include "logs/ilogs.hpp"

namespace tutil
{

struct NoSupportLogger final : public logs::iLogger
{
	/// Implementation of iLogger
	std::string get_log_level (void) const override { return ""; }

	/// Implementation of iLogger
	void set_log_level (const std::string& log_level) override {}

	/// Implementation of iLogger
	bool supports_level (size_t msg_level) const override { return false; }

	/// Implementation of iLogger
	bool supports_level (const std::string& msg_level) const override { return false; }

	/// Implementation of iLogger
	void log (size_t msg_level, const std::string& msg,
		const logs::SrcLocT& location = logs::SrcLocT::current()) override
	{
		called_ = true;
	}

	/// Implementation of iLogger
	void log (const std::string& msg_level, const std::string& msg,
		const logs::SrcLocT& location = logs::SrcLocT::current()) override
	{
		called_ = true;
	}

	bool called_ = false;
};

}

#endif // TEST_NOSUPPORT_HPP
