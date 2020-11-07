
#ifndef TEST_PORT_CHECK_HPP
#define TEST_PORT_CHECK_HPP

#include <boost/asio/io_service.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace tutil
{

bool port_in_use (unsigned short port);

}

#endif // TEST_PORT_CHECK_HPP
