a="Hello, World!"
table=str.maketrans("aeiou", "12345")
a=a.translate(table)
print(a)
table2=str.maketrans("12345", "aeiou")
a=a.translate(table2)
print(a)





































