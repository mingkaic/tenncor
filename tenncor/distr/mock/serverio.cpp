#include "tenncor/distr/mock/serverio.hpp"

#ifdef DISTR_MOCK_SERVERIO_HPP

types::StrUMapT<MockServer*> MockServerBuilder::mock_servers_;

std::mutex MockServerBuilder::mock_mtx_;

#endif
