/*!
 *
 *  tensor.hpp
 *  clay
 *
 *  Purpose:
 *  tensor manages data, shape, and type
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/shape.hpp"
#include "clay/dtype.hpp"

#include "clay/state.hpp"
#include "clay/isource.hpp"

#pragma once
#ifndef CLAY_ITENSOR_HPP
#define CLAY_ITENSOR_HPP

namespace clay
{

class iTensor
{
public:
	//! create a tensor of a specified shape
	iTensor (void) = default;

    virtual ~iTensor (void) = default;

    iTensor (const iTensor& other) = default;
    iTensor& operator = (const iTensor& other) = default;

    iTensor (iTensor&& other) = delete;
    iTensor& operator = (iTensor&& other) = delete;

    iTensor* clone (void) const
    {
        return clone_impl();
    }


	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	//! get internal state
	virtual State get_state (void) const = 0;

	//! get tensor shape
	virtual Shape get_shape (void) const = 0;

	//! get tensor dtype
	virtual DTYPE get_type (void) const = 0;

	//! get bytes allocated
	virtual size_t total_bytes (void) const = 0;


	// >>>>>>>>>>>> MUTATOR <<<<<<<<<<<<

	//! copy over data from src
	//! return true if successful
	bool read_from (const iSource& src)
    {
        State state = get_state();
        return src.read_data(state);
    }

protected:
    virtual iTensor* clone_impl (void) const = 0;
};

using TensorPtrT = std::unique_ptr<iTensor>;

}

#endif /* CLAY_ITENSOR_HPP */
