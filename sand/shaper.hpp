#include <functional>

#include "sand/shape.hpp"
#include "sand/opcode.hpp"

#ifndef SHAPER_HPP
#define SHAPER_HPP

using Shaper = std::function<Shape(std::vector<Shape>)>;

Shape elem_shaper (std::vector<Shape> shapes);

Shape transpose_shaper (std::vector<Shape> shapes);

Shape matmul_shaper (std::vector<Shape> shapes);

Shaper get_shaper (OPCODE opcode);

#endif /* SHAPER_HPP */
