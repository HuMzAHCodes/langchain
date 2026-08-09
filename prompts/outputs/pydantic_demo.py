from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):

    name: str = 'nitish'                    # has a default -> optional at construction
    age: Optional[int] = None               # may be int or None
    email: EmailStr                          # required + format-validated
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')


new_student = {'age': '32', 'email': 'abc@gmail.com'}   # note: age is a STRING here

student = Student(**new_student)             # dict unpacked into keyword args; validated on creation

student_dict = dict(student)                 # model -> plain dict

print(student_dict['age'])                   # prints 32 as an int, not '32'

student_json = student.model_dump_json()     # model -> JSON string


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Data Validation with Pydantic (BaseModel)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT PYDANTIC IS FOR
#   A TypedDict (previous file) only describes a shape — it does NOTHING at
#   runtime; Python won't stop you putting garbage in. Pydantic's BaseModel
#   actually VALIDATES and COERCES data when the object is created. If the data
#   is wrong, it raises ValidationError instead of silently letting bad values
#   through. This is why it's the go-to for LLM outputs, API request bodies, and
#   config