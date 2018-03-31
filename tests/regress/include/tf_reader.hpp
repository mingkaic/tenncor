#include <vector>
#include <string>

#include "include/graph/variable.hpp"

#pragma once
#ifndef TF_READER_HPP
#define TF_READER_HPP

std::vector<double> parse_param (std::string str);

nnet::variable* tensify (std::string str);

#endif /* TF_READER_HPP */
