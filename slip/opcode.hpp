/*!
 *
 *  opcode.hpp
 *  slip
 *
 *  Purpose:
 *  OPCODE enum definition
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <unordered_map>

#include "mold/functor.hpp"

#pragma once
#ifndef SLIP_OPCODE_HPP
#define SLIP_OPCODE_HPP

namespace slip
{

enum OPCODE
{
	CAST = 0,
	ABS,
	NEG,
	NOT,
	SIN,
	COS,
	TAN,
	EXP,
	LOG,
	SQRT,
	ROUND,
	POW,
	ADD,
	SUB,
	MUL,
	DIV,
	EQ,
	NE,
	GT,
	LT,
	BINO,
	UNIF,
	NORM,
	TRANSPOSE,
	FLIP,
	ARGMAX,
	RMAX,
	RSUM,
	EXPAND,
	N_ELEMS,
	N_DIMS,
	MATMUL,
	// gradient nodes (todo: remove this)
	INJACOBIAN,
	OUTJACOBIAN,
	JACOBIANLEFT,
	JACOBIANRIGHT,
	// sentinel
	_SENTINEL
};

struct EnumHash
{
	template <typename T>
	size_t operator() (T e) const
	{
		return static_cast<size_t>(e);
	}
};

using OpnameMap = std::unordered_map<OPCODE,std::string,EnumHash>;

extern OpnameMap opnames;

}

#endif /* SLIP_OPCODE_HPP */
