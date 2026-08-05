from langchain_openai import  ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


# 
# the language is ranged between 0 to 2 , it is the creativity parameter
# upto .3 is for facts 
# .9 to 1.2 for the creative generative text tasks 



# max_completion_tokens is the maximum outupt you need in the output 
# good for FINANCE MANAGEMENT
model=ChatOpenAI(model="gpt-4", language=1.2,max_completion_tokens=80)


result=model.invoke("what is the capital of iran")


print(result.content)


# // we use this ( chat models ) instead of llms