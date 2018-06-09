#include <vector>
#include <string>

#include "kiln/variable.hpp"

#pragma once
#ifndef TF_READER_HPP
#define TF_READER_HPP

std::vector<double> parse_param (std::string str);

kiln::Variable* varify (std::string str, std::string label);

#endif /* TF_READER_HPP */
