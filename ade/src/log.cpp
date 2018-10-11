#include <iostream>

#include "ade/log.hpp"

#ifdef ADE_LOG_HPP

namespace ade
{

static const std::string WARN_TAG = "[WARNING]:";
static const std::string ERR_TAG = "[ERROR]:";

struct DefLogger : public iLogger
{
	void warn (std::string msg) const override
	{
		std::cerr << ade::WARN_TAG << msg << std::endl;
	}

	void error (std::string msg) const override
	{
		std::cerr << ade::ERR_TAG << msg << std::endl;
	}

	void fatal (std::string msg) const override
	{
		throw std::runtime_error(msg);
	}
};

static std::shared_ptr<iLogger> glogger = std::make_shared<DefLogger>();

void set_logger (std::shared_ptr<iLogger> logger)
{
	glogger = logger;
}

const iLogger& get_logger (void)
{
	return *glogger;
}

void warn (std::string msg)
{
    get_logger().warn(msg);
}

void error (std::string msg)
{
    get_logger().error(msg);
}

void fatal (std::string msg)
{
    get_logger().fatal(msg);
}

}

#endif
