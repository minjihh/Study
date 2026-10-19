from typing import Dict, TypedDict

# Dict[str, str] -> type hint, 강제성은 없음
sample_dict : Dict[str, str] = {
    'name' : '김말똥',
    'age' : '39',
    'job' : '회사원',
} 

# print(sample_dict)
# {'name': '김말똥', 'age': '39', 'job': '회사원'}
# print(sample_dict['name'])

###########################################################3
class Person(TypedDict):
    name : str
    age : int
    job : str

person = Person(name='홍길동', age=40, job = '도적')
print(person)

###########################################################3

person2 : Person = {'name': '김개똥', 'age' : 20, 'job': '전사'}
print(person2)