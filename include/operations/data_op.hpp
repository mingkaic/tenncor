/*!
 *
 *  data_op.hpp
 *  cnnet
 *
 *  Purpose:
 *  performs numerical data on a void array
 *
 *  Created by Mingkai Chen on 2018-1-14.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_DATA_OP_HPP
#define TENNCOR_DATA_OP_HPP

namespace nnet
{

using VARR = std::pair<void*,tensorshape>;

using ARGS = std::vector<size_t>;

using VFUNC = std::function<void(VARR,std::vector<VARR>,ARGS)>;

void operate (std::string opname, TENS_TYPE type, VARR dest, std::vector<VARR> src, ARGS args);

#ifndef TENNCOR_D_UNARY_HPP
#define TENNCOR_D_UNARY_HPP

template <typename T>
void abs (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void neg (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void sin (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void cos (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void tan (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void csc (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void sec (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void cot (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void exp (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void ln (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void sqrt (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void round (VARR dest, std::vector<VARR> srcs, ARGS);

#endif /* TENNCOR_D_UNARY_HPP */

#ifndef TENNCOR_D_NNARY_HPP
#define TENNCOR_D_NNARY_HPP

template <typename T>
void clip (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void clip_norm (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void binom (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void pow (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void add (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void sub (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void mul (VARR dest, std::vector<VARR> srcs, ARGS);

template <typename T>
void div (VARR dest, std::vector<VARR> srcs, ARGS);

#endif /* TENNCOR_D_NNARY_HPP */

#ifndef TENNCOR_D_MATMUL_HPP
#define TENNCOR_D_MATMUL_HPP

#define STRASSEN_THRESHOLD 256

template <typename T>
void matmul (VARR dest, std::vector<VARR> srcs, ARGS);

#endif /* TENNCOR_D_MATMUL_HPP */

#ifndef TENNCOR_D_SHAPED_HPP
#define TENNCOR_D_SHAPED_HPP

template <typename T>
void extend (VARR dest, std::vector<VARR> srcs, ARGS args);

template <typename T>
void flip (VARR dest, std::vector<VARR> srcs, ARGS args);

template <typename T>
void crosscorr2d (VARR dest, std::vector<VARR> srcs, ARGS args);

#endif /* TENNCOR_D_SHAPED_HPP */

}

#include "src/operations/d_unary.ipp"

#include "src/operations/d_matmul.ipp"

#endif /* TENNCOR_DATA_OP_HPP */
