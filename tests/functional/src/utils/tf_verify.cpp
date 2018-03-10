#include <numeric>
#include <vector>
#include <string>

#include "tests/functional/include/tf_verify.hpp"
#include "tests/functional/include/tf_reader.hpp"

#ifdef TF_VERIFY_HPP

// #define EPSILON std::numeric_limits<double>::epsilon() // uncomment after protobuf check
#define EPSILON 0.00005 // relative error threshold

std::unordered_set<std::string> TF_VERIFY::read_files_;

std::unordered_map<std::string,std::string> TF_VERIFY::ops_;

op_args TF_VERIFY::parse_line (std::string opname)
{
	op_args result;
	std::string line = ops_[opname];
	size_t length;
	size_t start = 0;
	size_t end = line.find(',');
	// assert end != std::string::npos
	result.params_ = parse_param(line.substr(start, end));

	while (end != std::string::npos)
	{
		start = end + 1;
		end = line.find(',', start);
		if (end != std::string::npos)
		{
			length = end - start;
		}
		else
		{
			length = line.size() - start;
		}
		// assert that each tensor has aligned shape
		result.vars_.push_back(tensify(line.substr(start, length)));
	}
	return result;
}

void TF_VERIFY::to_mem (std::string file)
{
	if (read_files_.end() != read_files_.find(file))
	{
		return;
	}
	read_files_.insert(file);
	std::ifstream samplefile;
	char path[256];
	std::sprintf(path, "tests/functional/samples/%s.csv", file.c_str());
	samplefile.open(path);
	std::string line;
	if (samplefile.is_open())
	{
		while (std::getline(samplefile, line))
		{
			size_t start = 0;
			size_t end = line.find(',');
			std::string opname = line.substr(start, end);
			std::string remaining = line.substr(end+1);
			ops_[opname] = remaining;
		}
	}
	else
	{
		throw std::exception(); // can't open file
	}
	samplefile.close();
}

void vec_match (std::vector<size_t> expectv, std::vector<size_t> gotv)
{
	ASSERT_EQ(expectv.size(), gotv.size());
	for (size_t i = 0; i < expectv.size(); ++i)
	{
		ASSERT_EQ(expectv[i], gotv[i]);
	}
}

void vec_check (std::vector<double> expectv, std::vector<double> gotv)
{
	ASSERT_EQ(expectv.size(), gotv.size());
	for (size_t i = 0; i < expectv.size(); ++i)
	{
		double diff = std::abs(expectv[i] - gotv[i]);
		if (expectv[i] != 0)
		{
			diff /= expectv[i];
		}
		if (EPSILON < diff)
		{
			std::cout << i << " " << expectv[i] << " " << gotv[i] << std::endl;
		}
		ASSERT_GE(EPSILON, diff);
	}
}

void tensor_check (const nnet::tensor* expect, 
	const nnet::tensor* result)
{
	nnet::portal_dest portex;
	nnet::portal_dest portgo;
	expect->write_to(portex);
	result->write_to(portgo);
	assert(portex.type_ == DOUBLE && portgo.type_ == DOUBLE);
	std::vector<size_t> expects = portex.shape_.as_list();
	std::vector<size_t> gots = portgo.shape_.as_list();
	size_t exn = portex.shape_.n_elems();
	size_t gon = portgo.shape_.n_elems();
	assert(!portex.data_.expired());
	assert(!portgo.data_.expired());
	double* exptr = (double*) portex.data_.lock().get();
	double* goptr = (double*) portgo.data_.lock().get();
	std::vector<double> expectv(exptr, exptr + exn);
	std::vector<double> gotv(goptr, goptr + gon);

	vec_match(expects, gots);
	vec_check(expectv, gotv);
}

#endif
