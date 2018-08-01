/*!
 *
 *  stream.hpp
 *  ioutil
 *
 *  Purpose:
 *  define commonly used stream formatter
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <vector>
#include <sstream>
#include <iterator>

#pragma once
#ifndef IOUTIL_STREAM_HPP
#define IOUTIL_STREAM_HPP

namespace ioutil
{

class Stream final
{
public:
	//! to_stream conversion enum trick
	enum convert_to_string
	{
		to_str
	};

	Stream (void);

	//! no copying or moving to force string conversion
	//! format content is always passed as a string
	Stream (const Stream&) = delete;
	Stream (Stream&&) = delete;
	Stream& operator = (const Stream&) = delete;
	Stream& operator = (Stream&&) = delete;

	//! overload << operator for non-vector values to add to Stream
	template <typename T>
	Stream& operator << (const T& value)
	{
		stream_ << value;
		return *this;
	}

	//! overload << operator for vectors
	//! add values to string delimited by comma
	template <typename T>
	Stream& operator << (const std::vector<T>& values)
	{
		size_t n = values.size();
		if (n > 0)
		{
			stream_ << values[0];
			for (size_t i = 1; i < n; ++i)
			{
				stream_ << " " << values[i];
			}
		}
		return *this;
	}

	//! explicit string conversion function
	std::string str (void) const;

	//! implicit string converter
	operator std::string () const;

	//! out Stream as string
	std::string operator >> (convert_to_string);

private:
	std::stringstream stream_;
};

}

#endif /* IOUTIL_STREAM_HPP */
