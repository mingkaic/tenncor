%{
#include <stdio.h>
#include <string.h>

#include "ade_cli/ast/calc_ast.h"

#include "ade_cli/y.tab.h"

int c;

%}

%%
" "		;

abs		{
	yylval.int_type = ABS;
	return UNARY;
}

sin		{
	yylval.int_type = SIN;
	return UNARY;
}

cos		{
	yylval.int_type = COS;
	return UNARY;
}

tan		{
	yylval.int_type = TAN;
	return UNARY;
}

exp		{
	yylval.int_type = EXP;
	return UNARY;
}

log		{
	yylval.int_type = LOG;
	return UNARY;
}

sqrt	{
	yylval.int_type = SQRT;
	return UNARY;
}

round	{
	yylval.int_type = ROUND;
	return UNARY;
}

flip	{
	yylval.int_type = FLIP;
	return UNARY;
}

pow		{
	yylval.int_type = POW;
	return BINARY;
}

binomial	{
	yylval.int_type = BINO;
	return BINARY;
}

uniform		{
	yylval.int_type = UNIF;
	return BINARY;
}

normal		{
	yylval.int_type = NORM;
	return BINARY;
}

n_elems		{
	yylval.int_type = NELEMS;
	return UNARY;
}

n_dims		{
	yylval.int_type = NDIMS;
	return UNARY;
}

argmax	{
	yylval.int_type = ARGMAX;
	return UNARY;
}

rmax	{
	yylval.int_type = RMAX;
	return UNARY;
}

rsum	{
	yylval.int_type = RSUM;
	return UNARY;
}

permute	{
	yylval.int_type = PERMUTE;
	return SHAPEOP;
}

extend	{
	yylval.int_type = EXTEND;
	return SHAPEOP;
}

reshape	{
	yylval.int_type = RESHAPE;
	return SHAPEOP;
}

matmul	{
	yylval.int_type = MATMUL;
	return BINARY;
}

grad	{
	return GRAD;
}

print	{
	return SHOW_EQ;
}

exit 	{
	return EXIT;
}

[a-zA-Z\_][a-zA-Z\_0-9]{0,31}	{
	strncpy(yylval.str_type, yytext, 32);
	return VAR;
}

[0-9]	{
	c = yytext[0];
	yylval.int_type = c - '0';
	return DIGIT;
}

[^a-z0-9;\b]	{
	c = yytext[0];
	return c;
}
%%
