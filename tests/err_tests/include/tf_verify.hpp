#include <unordered_map>
#include <unordered_set>
#include <fstream>

#include "gtest/gtest.h"

#include "include/graph/leaf/variable.hpp"

#ifndef TF_VERIFY_HPP
#define TF_VERIFY_HPP

struct op_args
{
	std::vector<double> params_;

	std::vector<nnet::variable*> vars_;
};

class TF_VERIFY : public ::testing::Test
{
public:
	static std::unordered_set<std::string> read_files_;
	static std::unordered_map<std::string,std::string> ops_;

	op_args parse_line (std::string opname);

protected:
	static void to_mem (std::string file);

	virtual void SetUp (void) {}

	virtual void TearDown (void) {}
};

void vec_match (std::vector<size_t> expectv, std::vector<size_t> gotv);

void vec_check (std::vector<double> expectv, std::vector<double> gotv);

void tensor_check (const nnet::tensor* expect, 
	const nnet::tensor* result);

#endif /* TF_VERIFY_HPP */
