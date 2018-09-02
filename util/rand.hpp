#include <string>
#include <random>

#ifndef UTIL_RAND_HPP
#define UTIL_RAND_HPP

using ENGINE = std::default_random_engine;

ENGINE& get_engine (void);

std::string make_uid (void* ptr, ENGINE& engine = get_engine());

#endif /* UTIL_RAND_HPP */
