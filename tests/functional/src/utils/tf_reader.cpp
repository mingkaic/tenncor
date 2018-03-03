#include "tests/functional/include/tf_reader.hpp"

#ifdef TF_READER_HPP

std::vector<double> parse_param (std::string str)
{
	std::vector<double> param;
	size_t length, start;
	size_t end = -1;
	do
	{
		start = end + 1;
		end = str.find(' ', start);
		if (end != std::string::npos)
		{
			length = end - start;
		}
		else
		{
			length = str.size() - start;
		}
		std::string subs = str.substr(start, length);
		if (subs.size() > 0)
		{
			param.push_back(std::atof(subs.c_str()));
		}
	}
	while (end != std::string::npos);
	return param;
}

nnet::variable* tensify (std::string str)
{
	// get shape rank
	size_t rank = 0;
	while (rank < str.size() && '[' == str[rank])
	{
		++rank;
	}
	if (rank == 0)
	{
		std::shared_ptr<nnet::const_init> ci = std::make_shared<nnet::const_init>();
		ci->set((double) std::atof(str.c_str()));
		return new nnet::variable(nnet::tensorshape(std::vector<size_t>{1}), ci, "");
	}
	std::vector<double> data;
	std::vector<size_t> shaplist(rank, 0);

	{
		size_t max_level = 0;
		size_t level = 0;
		size_t count = 0;
		std::string buffer;
		for (size_t i = rank; i < str.size(); ++i)
		{
			if (str[i] == ' ' || str[i] == ']')
			{
				if (buffer.size() > 0)
				{
					double fdata = std::atof(buffer.c_str());
					data.push_back(fdata);
					if (max_level == 0)
					{
						// zeroth dimension is the count of numbers
						// subsequent dimensions count by delimiter ']'
						++count;
					}
				}
				buffer = "";
				if (str[i] == ']')
				{
					++level;
					if (level > max_level)
					{
						assert(level <= rank);
						// record the previous dimension
						shaplist[level - 1] = count;
						// start counting towards the new dimension
						max_level = level;
						count = 1;
					}
					else if (level == max_level)
					{
						// count towards the current dimension
						count++;
					}
				}
			}
			else if (str[i] == '[')
			{
				--level;
			}
			else
			{
				buffer.push_back(str[i]);
			}
		}
		// assert max_level == rank
	}

	size_t totalbytes = data.size() * sizeof(double);
	std::shared_ptr<void> sdata = nnutils::make_svoid(totalbytes);
	std::memcpy(sdata.get(), &data[0], totalbytes);
	std::shared_ptr<nnet::assign_io> asgn = std::make_shared<nnet::assign_io>();
	asgn->set_data(sdata, DOUBLE, nnet::tensorshape(shaplist), 0);

	return new nnet::variable(nnet::tensorshape(shaplist), asgn, "");
}

#endif
