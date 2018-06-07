/*!
 *
 *  pb_build.hpp
 *  lead
 *
 *  Purpose:
 *  protobuf build
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/ibuilder.hpp"

#include "lead/data.pb.h"
#include "lead/include/packer.hpp"

#pragma once
#ifndef LEAD_PB_BUILDER_HPP
#define LEAD_PB_BUILDER_HPP

namespace lead
{

class PbBuilder final : public clay::iBuilder
{
public:
	PbBuilder (const tenncor::TensorPb& pb);

	clay::TensorPtrT get (void) const override;

	clay::TensorPtrT get (clay::Shape shape) const override;

protected:
	clay::iBuilder* clone_impl (void) const override
	{
		return new PbBuilder(*this);
	}

private:
	tenncor::TensorPb pb_;
};

}

#endif /* LEAD_PB_BUILDER_HPP */
