#include <memory>

#ifndef TENNCOR_SHARED_PTR_HPP
#define TENNCOR_SHARED_PTR_HPP

namespace nnutils
{

std::shared_ptr<void> make_svoid (size_t nbytes);

void check_ptr (std::shared_ptr<void>& ptr, size_t nbytes);

}

#endif /* TENNCOR_SHARED_PTR_HPP */
