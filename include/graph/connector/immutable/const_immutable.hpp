/*!
 *
 *  const_immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  an immutable extension that
 *  clears all leaves and mimic constant behavior
 *
 *  Created by Mingkai Chen on 2017-09-17.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef CONST_IMMUTABLE_HPP
#define CONST_IMMUTABLE_HPP

namespace nnet
{

class const_immutable : public immutable
{
public:
	static const_immutable* get (inode* x);

private:
	const_immutable (inode* x);
};

}

#endif /* CONST_IMMUTABLE_HPP */
