#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

#include "soil/shape.hpp"
#include "soil/type.hpp"
#include "soil/error.hpp"
#include "soil/mapper.hpp"

#ifndef DATA_HPP
#define DATA_HPP

std::shared_ptr<char> make_data (size_t nbytes);

std::shared_ptr<char> make_data (char* data, size_t nbytes);

#endif /* DATA_HPP */
