%{
#include <string.h>

#include "cli/llo/ast/ast.h"

#include "cli/llo/y.tab.h"

int c;

%}

%%
[\t\r\a\v\b\n ]	;

=	{ return ASSIGN; }

\(	{ return LPAREN; }

\)	{ return RPAREN; }

,	{ return COMMA; }

\+	{ return PLUS; }

-	{ return MINUS; }

\*	{ return STAR; }

\/	{ return SLASH; }

\[ 	{ return LSB; }

\]	{ return RSB; }

;	{ return ENDSTMT; }

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
	return DIMOP;
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
	return DIMOP;
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

mode	{
	return MODE;
}

shape	{
	return SHAPE;
}

print	{
	return PRINT;
}

exit 	{
	return EXIT;
}

[a-zA-Z\_][a-zA-Z\_0-9]{0,31}	{
	strncpy(yylval.str_type, yytext, 32);
	return VAR;
}

-?(0|[1-9][0-9]*)\.[0-9]*	{
	yylval.num_type = atof(yytext);
	return DECIMAL;
}

-[1-9][0-9]*	{
	yylval.num_type = atoi(yytext);
	return DECIMAL;
}

0|[1-9][0-9]*	{
	yylval.int_type = atoi(yytext);
	return INTEGER;
}
%%
