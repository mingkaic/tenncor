/*!
 *
 *  const_con.hpp
 *  cnnet
 *
 *  Purpose:
 *  an connector extension that
 *  clears all leaves and mimic constant behavior
 *
 *  Created by Mingkai Chen on 2017-09-17.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/linear.hpp"

#pragma once
#define TENNCOR_CONST_CON_HPP
#ifndef TENNCOR_CONST_CON_HPP
#define TENNCOR_CONST_CON_HPP

namespace nnet
{

class const_con : public linear
{
public:
	static const_con* get (inode* x);

private:
	const_con (inode* x);
};

}

#endif /* TENNCOR_CONST_CON_HPP */
#undef TENNCOR_CONST_CON_HPP