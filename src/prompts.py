analysis_prompt = """### Please provide your understanding of the question, and be specific in identifying the subject corresponding to each value mentioned in the question.
# Question: {question}
# Understanding:"""

correct_prompt = """### Here is a sqlite SQL query that resulted from a question, but it produced an error when executed. Please correct it with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### Correct SQL: SELECT"""

reflect_prompt = """### Here is a sqlite SQL query that resulted from a question, but it produced an error when executed. What do you think is the possible reason for this SQL error?

### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### Error Reason:"""

sql_simple_prompt = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
SELECT"""

sql_middle_prompt = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
SELECT"""

sql_complex_prompt = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### Analysis for the question: {analysis}
SELECT"""

# ### Notice:
# 1. Don't try to use `LEFT JOIN` in your sqlite SQL query.
# 2. Don't select wrong tables or columns in your sqlite SQL query.
# 3. Don't use error foreign keys in your sqlite SQL query.
# 4. Whenever possible, try to avoid using `AND` or `OR` in your sqlite SQL query, you can use `INTERSECT` or `EXCEPT` instead.
# 5. The information from the `SELECT` cannot be unrelated to the information from the `GROUP BY`.
# 6. Try to use `ORDER BY` and `LIMIT 1 `to obtain extremum results.

correct_prompt_kg = """### Here is a sqlite SQL query that resulted from a question, but it produced an error when executed. Please correct it with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
{previous_information}
### Correct SQL: SELECT"""

reflect_prompt_kg = """### Here is a sqlite SQL query that resulted from a question, but it produced an error when executed. What do you think is the possible reason for this SQL error?

### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
{previous_information}
### Error Reason:"""

sql_simple_prompt_kg = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
SELECT"""

sql_middle_prompt_kg = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
SELECT"""

sql_complex_prompt_kg = """### Complete sqlite SQL query only and with no explanation.

### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### The explanation for database schema:
{explanation}
### Notice:
1. Don't try to use `LEFT JOIN` in your sqlite SQL query.
2. Don't select wrong tables or columns in your sqlite SQL query.
3. Don't use error foreign keys in your sqlite SQL query.
4. Whenever possible, try to avoid using `AND` or `OR` in your sqlite SQL query, you can use `INTERSECT` or `EXCEPT` instead.
5. The information from the `SELECT` cannot be unrelated to the information from the `GROUP BY`.
6. Try to use `ORDER BY` and `LIMIT 1 `to obtain extremum results.
### Question: {question}
### External Knowledge: {knowledge}
### Analysis for the question: {analysis}
SELECT"""

generate_without_anything_kg = """### This is the sqlite SQL query generated based on the question, its execution result.
Please judge its correctness based on the execution result and the explanation for the question.
If it's incorrect, output the correct sqlite SQL query; otherwise, output the original sqlite SQL query.
If the execution result is empty or 0, it is essentially due to issues in the original sqlite SQL query.

### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### SQLite SQL query: {SQL}
### Run results: {result}
SELECT"""

generate_without_anything = """### This is the sqlite SQL query generated based on the question, its execution result.
Please judge its correctness based on the execution result and the explanation for the question.
If it's incorrect, output the correct sqlite SQL query; otherwise, output the original sqlite SQL query.
If the execution result is empty or 0, it is essentially due to issues in the original sqlite SQL query.

### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### SQLite SQL query: {SQL}
### Run results: {result}
SELECT"""

schema_link_prompt = """### Here is the database information, questions. Please return the required table names and column names which can be used to answer the question in the form of a Python dictionary.
The keys should be table names, and the values should be column names. 
Please be aware of potential multiple-table join operations; obtaining the target values may require multiple intermediate tables or columns, and these contents should also be returned.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### Selected tables and columns:"""

schema_link_prompt_kg = """### Here is the database information, questions. Please return the required table names and column names which can be used to answer the question in the form of a Python dictionary.
The keys should be table names, and the values should be column names. 
Please be aware of potential multiple-table join operations; obtaining the target values may require multiple intermediate tables or columns, and these contents should also be returned.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### Selected tables and columns:"""


schema_rank_prompt = """### This is RankGPT, an intelligent assistant that can rank tables.columns based on their relevancy to the question.
### The following are {num} tables.columns, each indicated by number identifier []. I can rank them based on their relevance to question: {question}
#
{schema}
#
### The following are foreign keys.
#
{foreign_keys}
#
### The question is: {question}
### I will rank the {num} tables.columns above based on their relevance to the question. The tables.columns will be listed in descending order using identifiers, and the most relevant tables.columns should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
### Please note that I will consider utilizing foreign key relationships to select necessary columns.
The ranking results of the {num} tables.columns (only identifiers) is:"""


"""The ranking results of the {num} tables.columns (only identifiers) is:"""

schema_select_prompt = """### This is a ranking of tables.columns results based on their relevance to the question. Please truncate the results and make sure all the relevant ones have been retained.
### The following are {num} tables.columns, each indicated by number identifier [].
#
{schema}
#
### The following are foreign keys.
#
{foreign_keys}
#
### The question is: {question}
### The ranking result is: {ranking}
### The truncated result (only identifiers) is:"""


bias_eliminator_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
This represents the SQLite SQL query that has been generated in response to the given question, along with the resulting outcome after executing the query.
Please judge its correctness based on the execution result and the explanation for the question.
If it's incorrect, output the correct sqlite SQL query; otherwise, output the original sqlite SQL query.

### Input:
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### SQLite SQL query: {SQL}
### Run results: {result}

### Response:
"""

bias_eliminator_prompt_with_knowledge = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
This represents the SQLite SQL query that has been generated in response to the given question, along with the resulting outcome after executing the query.
Please judge its correctness based on the execution result and the explanation for the question.
If it's incorrect, output the correct sqlite SQL query; otherwise, output the original sqlite SQL query.

### Input:
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### SQLite SQL query: {SQL}
### Run results: {result}

### Response:
"""


find_error_reason = """### Here is the database information, questions, incorrectly generated SQL, and the standard correct SQL. Please select the corresponding category number based on the five types of errors. It may have more than one error categories.
Please ignore the case of table/column names.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### Incorrectly generated SQL: {error}
### Incorrectly result: {error_res}
### Standard correct SQL: {sql}
### Standard correct result: {res}
Category 1: Schema Linkage Error - selecting the wrong table, column, or using incorrect values.
Category 2: JOIN Error - joining the wrong table or columns.
Category 3: GROUP-BY Error - selecting the wrong column or not using GROUP-BY.
Category 4: Nesting Error - using the wrong operator or employing incorrect sub-queries.
Category 5: Others.

Error Category:"""