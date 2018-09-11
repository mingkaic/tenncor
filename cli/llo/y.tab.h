/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     INTEGER = 258,
     DECIMAL = 259,
     VAR = 260,
     UNARY = 261,
     BINARY = 262,
     DIMOP = 263,
     SHAPEOP = 264,
     GRAD = 265,
     PRINT = 266,
     SHAPE = 267,
     EXIT = 268,
     MODE = 269,
     ASSIGN = 270,
     LPAREN = 271,
     RPAREN = 272,
     LSB = 273,
     RSB = 274,
     COMMA = 275,
     PLUS = 276,
     MINUS = 277,
     STAR = 278,
     SLASH = 279,
     NEWLINE = 280,
     UMINUS = 281
   };
#endif
/* Tokens.  */
#define INTEGER 258
#define DECIMAL 259
#define VAR 260
#define UNARY 261
#define BINARY 262
#define DIMOP 263
#define SHAPEOP 264
#define GRAD 265
#define PRINT 266
#define SHAPE 267
#define EXIT 268
#define MODE 269
#define ASSIGN 270
#define LPAREN 271
#define RPAREN 272
#define LSB 273
#define RSB 274
#define COMMA 275
#define PLUS 276
#define MINUS 277
#define STAR 278
#define SLASH 279
#define NEWLINE 280
#define UMINUS 281




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 16 "calc.yacc"
{
	int int_type;
	double num_type;
	char str_type[32];
	struct DataHolder* data_type;
	struct ShapeHolder* vec_type;
	struct ASTNode* ref_type;
}
/* Line 1529 of yacc.c.  */
#line 110 "y.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

