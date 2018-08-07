#include <string>
#include <random>

#ifndef RAND_HPP
#define RAND_HPP

using ENGINE = std::default_random_engine;

ENGINE& get_engine (void);

std::string make_uid (void* ptr);

#endif /* RAND_HPP */
