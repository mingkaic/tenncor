#include <stdio.h>

#include "opt/parse/list.h"

#ifndef PARSE_DEF_H
#define PARSE_DEF_H

struct Branch
{
	int is_group_;
	struct PtrList* args_;
	char label_[32];
	char variadic_[32];
};

enum SUBGRAPH_TYPE
{
	SCALAR = 0,
	ANY,
	BRANCH,
};

struct Subgraph
{
	enum SUBGRAPH_TYPE type_;
	union
	{
		double scalar_;
		char* any_;
		struct Branch* branch_;
	} val_;
};

void subgraph_recursive_free (void* ptr);

struct Arg
{
	struct Subgraph* subgraph_;
	struct NumList* shaper_;
	struct NumList* coorder_;
};

void arg_recursive_free (void* ptr);

struct Conversion
{
	struct Subgraph* source_;
	struct Subgraph* dest_;
};

void conversion_recursive_free (void* ptr);

struct Group
{
	char ref_[32];
	char tag_[32];
};

struct Property
{
	char label_[32];
	char property_[32];
	int is_group_;
};

enum STMT_TYPE
{
	SYMBOL_DEF = 0,
	GROUP_DEF,
	PROPERTY_DEF,
	CONVERSION,
};

struct Statement
{
	enum STMT_TYPE type_;
	void* val_;
};

enum PTR_TYPE
{
	STATEMENT = 0,
	ARGUMENT,
};

void statement_recursive_free (void* ptr);

#ifdef __cplusplus
extern "C" {
#endif

// wrapper around Statement recursive free
extern void statements_free (struct PtrList* stmts);

extern int parse_str (struct PtrList** stmts, const char* str);

extern int parse_file (struct PtrList** stmts, FILE* file);

#ifdef __cplusplus
}
#endif

#endif // PARSE_DEF_H
