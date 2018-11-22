#include "testutil/common.hpp"

std::string TestLogger::latest_warning_;
std::string TestLogger::latest_error_;

std::shared_ptr<TestLogger> tlogger = std::make_shared<TestLogger>();
