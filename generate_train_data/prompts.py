generate_question_answer_without_knowledge = """### Generate a similar question and the corresponding gold SQL based on the given question and gold SQL.
Your output format should include "New Question" and "New Gold SQL"
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### Gold SQL: {sql}"""

generate_question_answer_with_knowledge = """### Generate a similar question, knowledge and the corresponding gold SQL based on the given question, external knowledge and gold SQL.
Your output format should include "New Question", "New External Knowledge" and "New Gold SQL"
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
### External Knowledge: {knowledge}
### Gold SQL: {sql}"""

prompt_with_knowledge_result = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

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

prompt_without_knowledge_result = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

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

sql_simple_prompt = """### Complete sqlite SQL query only and with no explanation.
### Sqlite SQL tables, with their properties:
#
{schema}
#
{foreign_keys}
#
### Question: {question}
SELECT"""