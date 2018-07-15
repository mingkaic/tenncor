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
	// ==== dimensionless ====
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

	// explicit dimension args
	ISMAX,

	FLIP,
	EXPAND,
	TRANSPOSE,

	// shaped
	N_ELEMS,
	N_DIMS, // explict args

	// ==== dimensioned ====
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

	// dimensioned (merge)
	ARGMAX,
	RMAX,
	RSUM,

	// dimensioned (assert 2 dimensions specified)
	MATMUL,

	// ==== deprecate ====
	// gradient nodes (todo: revise these)
	RESHAPE,
	JACOBIAN,
	TRACE_EXPAND,

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

template <typename K, typename V>
using EnumMap = std::unordered_map<K,V,EnumHash>;

using OpnameMap = EnumMap<OPCODE,std::string>;

extern const OpnameMap opnames;

}

#endif /* SLIP_OPCODE_HPP */
