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

/// Length of symbols
#define NSYMBOL 32

#define FALSE 0
#define TRUE 1

/// Represent map key-value pair
struct KeyVal
{
	int val_scalar_;
	struct NumList val_;
	char key_[NSYMBOL];
};

/// Recursively free the key-value pair and all it's contents
void kv_recursive_free (void* kv);

/// Argument for a branching node
struct Arg
{
	/// TreeNode node
	struct TreeNode* node_;
	/// Argument attributes
	struct PtrList attrs_;
};

/// Recursively free the argument and all it's contents
void arg_recursive_free (void* arg);

/// Branching (TreeNode or Group) node
struct Functor
{
	struct PtrList args_;
	char name_[NSYMBOL];
	char variadic_[NSYMBOL];
	int commutative_;
};

/// Recursively free the functor and all it's contents
void func_recursive_free (void* arg);

/// Rule tree node
struct TreeNode
{
	/// Rule tree node type
	enum SUBGRAPH_TYPE
	{
		SCALAR = 0, /// scalar_
		ANY, 		/// any_
		FUNCTOR, 	/// functor_
	};

	/// Type of the node
	enum SUBGRAPH_TYPE type_;
	/// Node instance according to type
	union
	{
		double scalar_;
		char* any_;
		struct Functor* functor_;
	} val_;
};

/// Recursively free the subgraph and all of its descendants
void node_recursive_free (void* node);

/// Represent a conversion between two rule trees
struct Conversion
{
	/// Root of matching subgraph
	struct TreeNode* matcher_;

	/// Root of target subgraph
	struct TreeNode* target_;
};

/// Recursively free the conversion and its sub nodes
void conversion_recursive_free (void* conv);

/// Pointer enumeration to store in PtrList
enum PTR_TYPE
{
	CONVERSION = 0,
	ARGUMENT,
	KV_PAIR,
};

void objs_free (struct PtrList* objs);

void objs_clear (struct PtrList* objs);

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
