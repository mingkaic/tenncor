#include <memory>

#ifndef DATA_HPP
#define DATA_HPP

std::shared_ptr<char> make_data (size_t nbytes);

std::shared_ptr<char> make_data (char* data, size_t nbytes);

#endif /* DATA_HPP */
