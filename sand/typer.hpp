#include <functional>

#include "sand/type.hpp"
#include "sand/opcode.hpp"

#ifndef TYPER_HPP
#define TYPER_HPP

using Typer = std::function<DTYPE(std::vector<DTYPE>)>;

DTYPE reg_typer (std::vector<DTYPE> args);

Typer get_typer (OPCODE opcode);

#endif /* TYPER_HPP */
