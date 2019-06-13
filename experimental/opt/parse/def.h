#include <stdlib.h>

#ifndef PARSE_DEF_HPP
#define PARSE_DEF_HPP

struct NumList
{
	struct NumList* next_;
	double val_;
};

struct ArgList;

struct Branch
{
	int is_group_;
	struct ArgList* args_;
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

struct Arg
{
	struct Subgraph* subgraph_;
	struct NumList* shaper_;
	struct NumList* coorder_;
};

struct ArgList
{
	struct ArgList* next_;
	struct Arg* val_;
};

struct Conversion
{
	struct Subgraph* source_;
	struct Subgraph* dest_;
};

enum STMT_TYPE
{
	SYMBOL_DEF = 0,
	CONVERSION,
};

struct StmtList
{
	struct StmtList* next_;
	enum STMT_TYPE type_;
	void* val_;
};

void list_recursive_free (struct NumList* list);

void subgraph_recursive_free (struct Subgraph* sg);

void arg_recursive_free (struct Arg* arg);

void arglist_recursive_free (struct ArgList* arglist);

void conversion_recursive_free (struct Conversion* conv);

extern void stmts_recursive_free (struct StmtList* stmts);

extern int parse_rule (struct StmtList** stmts, const char* filename);

#endif // PARSE_DEF_HPP
