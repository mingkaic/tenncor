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

#include "kiln/validator.hpp"

#include "clay/ibuilder.hpp"

#pragma once
#ifndef KILN_BUILDER_HPP
#define KILN_BUILDER_HPP

namespace kiln
{

class Builder : public clay::iBuilder
{
public:
	Builder (void) = default;

	Builder (Validator validate, clay::DTYPE dtype = clay::DTYPE::BAD);

	std::unique_ptr<clay::Tensor> get (void) const override;

	std::unique_ptr<clay::Tensor> get (clay::Shape shape) const override;

protected:
	virtual void init (char* dest, size_t nbytes) const = 0;

	clay::DTYPE dtype_ = clay::DTYPE::BAD;

private:
	Validator validate_;
};

}

#endif /* KILN_BUILDER_HPP */
