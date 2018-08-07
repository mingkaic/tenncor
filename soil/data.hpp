#include <memory>

#ifndef SOIL_DATA_HPP
#define SOIL_DATA_HPP

std::shared_ptr<char> make_data (size_t nbytes);

std::shared_ptr<char> make_data (char* data, size_t nbytes);

#endif /* SOIL_DATA_HPP */
