#include "testutil/port_check.hpp"

#ifdef TEST_PORT_CHECK_HPP

namespace tutil
{

bool port_in_use (unsigned short port)
{
	using namespace boost::asio;
	using ip::tcp;

	boost::asio::io_service svc;
	boost::asio::ip::tcp::acceptor a(svc);

	boost::system::error_code ec;
	a.open(boost::asio::ip::tcp::v4(), ec) || a.bind({
		boost::asio::ip::tcp::v4(), port
	}, ec);

	return ec == boost::asio::error::address_in_use;
}

}

#endif
