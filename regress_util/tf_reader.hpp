#include <vector>
#include <string>

#include "wire/variable.hpp"

#pragma once
#ifndef TF_READER_HPP
#define TF_READER_HPP

std::vector<double> parse_param (std::string str);

wire::Variable* varify (std::string str, std::string label);

#endif /* TF_READER_HPP */
