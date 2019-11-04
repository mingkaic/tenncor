///
/// def.h
/// opt/parse
///
/// Purpose:
/// Define c rule tree nodes and meta-data
///

#include <stdio.h>

#include "opt/parse/list.h"

#ifndef PARSE_DEF_H
#define PARSE_DEF_H

/// Branching (Functor or Group) node
struct Branch
{
	int is_group_;
	struct PtrList* args_;
	char label_[32];
	char variadic_[32];
};

/// Rule tree node type
enum SUBGRAPH_TYPE
{
	/// Definitive scalar constant
	SCALAR = 0,
	/// Rule tree leaf that represents any real node
	ANY,
	/// Branching node
	BRANCH,
};

/// Rule tree node
struct Subgraph
{
	/// Type of the node
	enum SUBGRAPH_TYPE type_;
	/// Node instance according to type
	union
	{
		double scalar_;
		char* any_;
		struct Branch* branch_;
	} val_;
};

/// Recursively free the subgraph and all of its descendants
void subgraph_recursive_free (struct Subgraph* sg);

/// Argument for a branching node
struct Arg
{
	/// Subgraph node
	struct Subgraph* subgraph_;

	/// Shape mapping meta-data
	struct NumList* shaper_;

	/// Coordinate mapping meta-data
	struct NumList* coorder_;
};

/// Recursively free the argument and all it's contents
void arg_recursive_free (void* arg);

/// Represent a conversion between two rule trees
struct Conversion
{
	/// Rule tree to target for conversion
	struct Subgraph* source_;

	/// Rule tree to convert to given the source tree's identified ANY children
	struct Subgraph* dest_;
};

/// Recursively free the conversion and its subgraphs
void conversion_recursive_free (void* conv);

/// Property association between label that
/// appears in rule tree and tag in real node
struct Property
{
	/// Functor/group label as it appears in rule tree
	char label_[32];

	/// Property of the group/functor
	char property_[32];

	/// Is 1 if the label refers to a group, otherwise it's functor
	int is_group_;
};

/// Statement type enum
enum STMT_TYPE
{
	/// Symbol definition (declaration of ANY nodes)
	SYMBOL_DEF = 0,
	/// Property association statement
	PROPERTY_DEF,
	/// Rule tree conversion statement
	CONVERSION,
};

/// Generic statement representation
struct Statement
{
	/// Statement type
	enum STMT_TYPE type_;

	/// type_=SYMBOL_DEF: val_ is a string
	/// type_=PROPERTY_DEF: val_ is struct Property*
	/// type_=CONVERSION: val_ is struct Conversion*
	void* val_;
};

/// Pointer enumeration to store in PtrList
enum PTR_TYPE
{
	STATEMENT = 0,
	ARGUMENT,
};

/// Safely free statement and it's contents recursively
/// Statement::type_=SYMBOL_DEF: free string
/// Statement::type_=PROPERTY_DEF: free property
/// Statement::type_=CONVERSION: conversion_recursive_free conversion
void statement_recursive_free (void* ptr);

#ifdef __cplusplus
extern "C" {
#endif

/// Recursively free all statements and their contents
extern void statements_free (struct PtrList* stmts);

/// Return 0 if successfully populate statements from parsed string
extern int parse_str (struct PtrList** stmts, const char* str);

/// Return 0 if successfully populate statements from parsed file
extern int parse_file (struct PtrList** stmts, FILE* file);

#ifdef __cplusplus
}
#endif

#endif // PARSE_DEF_H
