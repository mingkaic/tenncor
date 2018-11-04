#include "err/log.hpp"

#ifdef ERR_LOG_HPP

namespace err
{

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
