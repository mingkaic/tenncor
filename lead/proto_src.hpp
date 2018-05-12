/*!
 *
 *  builder.hpp
 *  kiln
 *
 *  Purpose:
 *  abstract builder that validates type and shape
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/ibuilder.hpp"

#pragma once
#ifndef LEAD_PB_BUILDER_HPP
#define LEAD_PB_BUILDER_HPP

namespace lead
{

class PbBuilder : public clay::iBuilder
{
public:
    PbBuilder (const lead::TensorPb& pb) :
        pb_(pb) {}

    clay::Tensor* get (void) const
    {

    }

    clay::Tensor* get (clay::Shape shape) const
    {
        
    }

private:
    lead::TensorPb pb_;
};

}

#endif /* LEAD_PB_BUILDER_HPP */